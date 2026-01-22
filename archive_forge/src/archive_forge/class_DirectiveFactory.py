from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
@six.add_metaclass(DirectiveFactoryMeta)
class DirectiveFactory(object):
    """Base for classes that provide a set of template directives.
    
    :since: version 0.6
    """
    directives = []
    'A list of ``(name, cls)`` tuples that define the set of directives\n    provided by this factory.\n    '

    def get_directive(self, name):
        """Return the directive class for the given name.
        
        :param name: the directive name as used in the template
        :return: the directive class
        :see: `Directive`
        """
        return self._dir_by_name.get(name)

    def get_directive_index(self, dir_cls):
        """Return a key for the given directive class that should be used to
        sort it among other directives on the same `SUB` event.
        
        The default implementation simply returns the index of the directive in
        the `directives` list.
        
        :param dir_cls: the directive class
        :return: the sort key
        """
        if dir_cls in self._dir_order:
            return self._dir_order.index(dir_cls)
        return len(self._dir_order)