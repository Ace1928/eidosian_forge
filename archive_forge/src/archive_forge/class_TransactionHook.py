import builtins
import types
import sys
from inspect import getmembers
from webob.exc import HTTPFound
from .util import iscontroller, _cfg
class TransactionHook(PecanHook):
    """
    :param start: A callable that will bind to a writable database and
                  start a transaction.
    :param start_ro: A callable that will bind to a readable database.
    :param commit: A callable that will commit the active transaction.
    :param rollback: A callable that will roll back the active
                     transaction.
    :param clear: A callable that will clear your current context.

    A basic framework hook for supporting wrapping requests in
    transactions. By default, it will wrap all but ``GET`` and ``HEAD``
    requests in a transaction. Override the ``is_transactional`` method
    to define your own rules for what requests should be transactional.
    """

    def __init__(self, start, start_ro, commit, rollback, clear):
        self.start = start
        self.start_ro = start_ro
        self.commit = commit
        self.rollback = rollback
        self.clear = clear

    def is_transactional(self, state):
        """
        Decide if a request should be wrapped in a transaction, based
        upon the state of the request. By default, wraps all but ``GET``
        and ``HEAD`` requests in a transaction, along with respecting
        the ``transactional`` decorator from :mod:pecan.decorators.

        :param state: The Pecan state object for the current request.
        """
        controller = getattr(state, 'controller', None)
        if controller:
            force_transactional = _cfg(controller).get('transactional', False)
        else:
            force_transactional = False
        if state.request.method not in ('GET', 'HEAD') or force_transactional:
            return True
        return False

    def on_route(self, state):
        state.request.error = False
        if self.is_transactional(state):
            state.request.transactional = True
            self.start()
        else:
            state.request.transactional = False
            self.start_ro()

    def before(self, state):
        if self.is_transactional(state) and (not getattr(state.request, 'transactional', False)):
            self.clear()
            state.request.transactional = True
            self.start()

    def on_error(self, state, e):
        trans_ignore_redirects = state.request.method not in ('GET', 'HEAD')
        if state.controller is not None:
            trans_ignore_redirects = _cfg(state.controller).get('transactional_ignore_redirects', trans_ignore_redirects)
        if type(e) is HTTPFound and trans_ignore_redirects is True:
            return
        state.request.error = True

    def after(self, state):
        if getattr(state.request, 'transactional', False):
            action_name = None
            if state.request.error:
                action_name = 'after_rollback'
                self.rollback()
            else:
                action_name = 'after_commit'
                self.commit()
            if action_name:
                controller = getattr(state, 'controller', None)
                if controller is not None:
                    actions = _cfg(controller).get(action_name, [])
                    for action in actions:
                        action()
        self.clear()