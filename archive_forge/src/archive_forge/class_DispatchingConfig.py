import re
import threading
class DispatchingConfig:
    """
    This is a configuration object that can be used globally,
    imported, have references held onto.  The configuration may differ
    by thread (or may not).

    Specific configurations are registered (and deregistered) either
    for the process or for threads.
    """
    _constructor_lock = threading.Lock()

    def __init__(self):
        self._constructor_lock.acquire()
        try:
            self.dispatching_id = 0
            while 1:
                self._local_key = 'paste.processconfig_%i' % self.dispatching_id
                if self._local_key not in local_dict():
                    break
                self.dispatching_id += 1
        finally:
            self._constructor_lock.release()
        self._process_configs = []

    def push_thread_config(self, conf):
        """
        Make ``conf`` the active configuration for this thread.
        Thread-local configuration always overrides process-wide
        configuration.

        This should be used like::

            conf = make_conf()
            dispatching_config.push_thread_config(conf)
            try:
                ... do stuff ...
            finally:
                dispatching_config.pop_thread_config(conf)
        """
        local_dict().setdefault(self._local_key, []).append(conf)

    def pop_thread_config(self, conf=None):
        """
        Remove a thread-local configuration.  If ``conf`` is given,
        it is checked against the popped configuration and an error
        is emitted if they don't match.
        """
        self._pop_from(local_dict()[self._local_key], conf)

    def _pop_from(self, lst, conf):
        popped = lst.pop()
        if conf is not None and popped is not conf:
            raise AssertionError(f'The config popped ({popped}) is not the same as the config expected ({conf})')

    def push_process_config(self, conf):
        """
        Like push_thread_config, but applies the configuration to
        the entire process.
        """
        self._process_configs.append(conf)

    def pop_process_config(self, conf=None):
        self._pop_from(self._process_configs, conf)

    def __getattr__(self, attr):
        conf = self.current_conf()
        if conf is None:
            raise AttributeError('No configuration has been registered for this process or thread')
        return getattr(conf, attr)

    def current_conf(self):
        thread_configs = local_dict().get(self._local_key)
        if thread_configs:
            return thread_configs[-1]
        elif self._process_configs:
            return self._process_configs[-1]
        else:
            return None

    def __getitem__(self, key):
        conf = self.current_conf()
        if conf is None:
            raise TypeError('No configuration has been registered for this process or thread')
        return conf[key]

    def __contains__(self, key):
        return key in self

    def __setitem__(self, key, value):
        conf = self.current_conf()
        conf[key] = value