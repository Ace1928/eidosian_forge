from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
class KNSpaceBehavior(object):
    """Inheriting from this class allows naming of the inherited objects, which
    are then added to the associated namespace :attr:`knspace` and accessible
    through it.

    Please see the :mod:`knspace behaviors module <kivy.uix.behaviors.knspace>`
    documentation for more information.
    """
    _knspace = ObjectProperty(None, allownone=True)
    _knsname = StringProperty('')
    __last_knspace = None
    __callbacks = None

    def __init__(self, knspace=None, **kwargs):
        self.knspace = knspace
        super(KNSpaceBehavior, self).__init__(**kwargs)

    def __knspace_clear_callbacks(self, *largs):
        for obj, name, uid in self.__callbacks:
            obj.unbind_uid(name, uid)
        last = self.__last_knspace
        self.__last_knspace = self.__callbacks = None
        assert self._knspace is None
        assert last
        new = self.__set_parent_knspace()
        if new is last:
            return
        self.property('_knspace').dispatch(self)
        name = self.knsname
        if not name:
            return
        if getattr(last, name) == self:
            setattr(last, name, None)
        if new:
            setattr(new, name, self)
        else:
            raise ValueError('Object has name "{}", but no namespace'.format(name))

    def __set_parent_knspace(self):
        callbacks = self.__callbacks = []
        fbind = self.fbind
        append = callbacks.append
        parent_key = self.knspace_key
        clear = self.__knspace_clear_callbacks
        append((self, 'knspace_key', fbind('knspace_key', clear)))
        if not parent_key:
            self.__last_knspace = knspace
            return knspace
        append((self, parent_key, fbind(parent_key, clear)))
        parent = getattr(self, parent_key, None)
        while parent is not None:
            fbind = parent.fbind
            parent_knspace = getattr(parent, 'knspace', 0)
            if parent_knspace != 0:
                append((parent, 'knspace', fbind('knspace', clear)))
                self.__last_knspace = parent_knspace
                return parent_knspace
            append((parent, parent_key, fbind(parent_key, clear)))
            new_parent = getattr(parent, parent_key, None)
            if new_parent is parent:
                break
            parent = new_parent
        self.__last_knspace = knspace
        return knspace

    def _get_knspace(self):
        _knspace = self._knspace
        if _knspace is not None:
            return _knspace
        if self.__callbacks is not None:
            return self.__last_knspace
        return self.__set_parent_knspace()

    def _set_knspace(self, value):
        if value is self._knspace:
            return
        knspace = self._knspace or self.__last_knspace
        name = self.knsname
        if name and knspace and (getattr(knspace, name) == self):
            setattr(knspace, name, None)
        if value == 'fork':
            if not knspace:
                knspace = self.knspace
            if knspace:
                value = knspace.fork()
            else:
                raise ValueError('Cannot fork with no namespace')
        for obj, prop_name, uid in self.__callbacks or []:
            obj.unbind_uid(prop_name, uid)
        self.__last_knspace = self.__callbacks = None
        if name:
            if value is None:
                knspace = self.__set_parent_knspace()
                if knspace:
                    setattr(knspace, name, self)
                self._knspace = None
            else:
                setattr(value, name, self)
                knspace = self._knspace = value
            if not knspace:
                raise ValueError('Object has name "{}", but no namespace'.format(name))
        else:
            if value is None:
                self.__set_parent_knspace()
            self._knspace = value
    knspace = AliasProperty(_get_knspace, _set_knspace, bind=('_knspace',), cache=False, rebind=True, allownone=True)
    "The namespace instance, :class:`KNSpace`, associated with this widget.\n    The :attr:`knspace` namespace stores this widget when naming this widget\n    with :attr:`knsname`.\n\n    If the namespace has been set with a :class:`KNSpace` instance, e.g. with\n    `self.knspace = KNSpace()`, then that instance is returned (setting with\n    `None` doesn't count). Otherwise, if :attr:`knspace_key` is not None, we\n    look for a namespace to use in the object that is stored in the property\n    named :attr:`knspace_key`, of this instance. I.e.\n    `object = getattr(self, self.knspace_key)`.\n\n    If that object has a knspace property, then we return its value. Otherwise,\n    we go further up, e.g. with `getattr(object, self.knspace_key)` and look\n    for its `knspace` property.\n\n    Finally, if we reach a value of `None`, or :attr:`knspace_key` was `None`,\n    the default :attr:`~kivy.uix.behaviors.knspace.knspace` namespace is\n    returned.\n\n    If :attr:`knspace` is set to the string `'fork'`, the current namespace\n    in :attr:`knspace` will be forked with :meth:`KNSpace.fork` and the\n    resulting namespace will be assigned to this instance's :attr:`knspace`.\n    See the module examples for a motivating example.\n\n    Both `rebind` and `allownone` are `True`.\n    "
    knspace_key = StringProperty('parent', allownone=True)
    "The name of the property of this instance, to use to search upwards for\n    a namespace to use by this instance. Defaults to `'parent'` so that we'll\n    search the parent tree. See :attr:`knspace`.\n\n    When `None`, we won't search the parent tree for the namespace.\n    `allownone` is `True`.\n    "

    def _get_knsname(self):
        return self._knsname

    def _set_knsname(self, value):
        old_name = self._knsname
        knspace = self.knspace
        if old_name and knspace and (getattr(knspace, old_name) == self):
            setattr(knspace, old_name, None)
        self._knsname = value
        if value:
            if knspace:
                setattr(knspace, value, self)
            else:
                raise ValueError('Object has name "{}", but no namespace'.format(value))
    knsname = AliasProperty(_get_knsname, _set_knsname, bind=('_knsname',), cache=False)
    'The name given to this instance. If named, the name will be added to the\n    associated :attr:`knspace` namespace, which will then point to the\n    `proxy_ref` of this instance.\n\n    When named, one can access this object by e.g. self.knspace.name, where\n    `name` is the given name of this instance. See :attr:`knspace` and the\n    module description for more details.\n    '