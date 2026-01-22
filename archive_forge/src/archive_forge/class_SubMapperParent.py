import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
class SubMapperParent(object):
    """Base class for Mapper and SubMapper, both of which may be the parent
    of SubMapper objects
    """

    def submapper(self, **kargs):
        """Create a partial version of the Mapper with the designated
        options set

        This results in a :class:`routes.mapper.SubMapper` object.

        If keyword arguments provided to this method also exist in the
        keyword arguments provided to the submapper, their values will
        be merged with the saved options going first.

        In addition to :class:`routes.route.Route` arguments, submapper
        can also take a ``path_prefix`` argument which will be
        prepended to the path of all routes that are connected.

        Example::

            >>> map = Mapper(controller_scan=None)
            >>> map.connect('home', '/', controller='home', action='splash')
            >>> map.matchlist[0].name == 'home'
            True
            >>> m = map.submapper(controller='home')
            >>> m.connect('index', '/index', action='index')
            >>> map.matchlist[1].name == 'index'
            True
            >>> map.matchlist[1].defaults['controller'] == 'home'
            True

        Optional ``collection_name`` and ``resource_name`` arguments are
        used in the generation of route names by the ``action`` and
        ``link`` methods.  These in turn are used by the ``index``,
        ``new``, ``create``, ``show``, ``edit``, ``update`` and
        ``delete`` methods which may be invoked indirectly by listing
        them in the ``actions`` argument.  If the ``formatted`` argument
        is set to ``True`` (the default), generated paths are given the
        suffix '{.format}' which matches or generates an optional format
        extension.

        Example::

            >>> from routes.util import url_for
            >>> map = Mapper(controller_scan=None)
            >>> m = map.submapper(path_prefix='/entries', collection_name='entries', resource_name='entry', actions=['index', 'new'])
            >>> url_for('entries') == '/entries'
            True
            >>> url_for('new_entry', format='xml') == '/entries/new.xml'
            True

        """
        return SubMapper(self, **kargs)

    def collection(self, collection_name, resource_name, path_prefix=None, member_prefix='/{id}', controller=None, collection_actions=COLLECTION_ACTIONS, member_actions=MEMBER_ACTIONS, member_options=None, **kwargs):
        """Create a submapper that represents a collection.

        This results in a :class:`routes.mapper.SubMapper` object, with a
        ``member`` property of the same type that represents the collection's
        member resources.

        Its interface is the same as the ``submapper`` together with
        ``member_prefix``, ``member_actions`` and ``member_options``
        which are passed to the ``member`` submapper as ``path_prefix``,
        ``actions`` and keyword arguments respectively.

        Example::

            >>> from routes.util import url_for
            >>> map = Mapper(controller_scan=None)
            >>> c = map.collection('entries', 'entry')
            >>> c.member.link('ping', method='POST')
            >>> url_for('entries') == '/entries'
            True
            >>> url_for('edit_entry', id=1) == '/entries/1/edit'
            True
            >>> url_for('ping_entry', id=1) == '/entries/1/ping'
            True

        """
        if controller is None:
            controller = resource_name or collection_name
        if path_prefix is None:
            if collection_name is None:
                path_prefix_str = ''
            else:
                path_prefix_str = '/{collection_name}'
        elif collection_name is None:
            path_prefix_str = '{pre}'
        else:
            path_prefix_str = '{pre}/{collection_name}'
        path_prefix = path_prefix_str.format(pre=path_prefix, collection_name=collection_name)
        collection = SubMapper(self, collection_name=collection_name, resource_name=resource_name, path_prefix=path_prefix, controller=controller, actions=collection_actions, **kwargs)
        collection.member = SubMapper(collection, path_prefix=member_prefix, actions=member_actions, **member_options or {})
        return collection