import copy
import threading
import contextlib
import operator
import copyreg
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Iterable, Generic, TYPE_CHECKING
class ProxyObjectV2(Generic[ProxyObjT]):

    def __init__(self, obj_cls: Optional[Union[Type[ProxyObjT], str]]=None, obj_getter: Optional[Union[Callable, str]]=None, obj_args: Optional[List[Any]]=None, obj_kwargs: Optional[Dict[str, Any]]=None, obj_initialize: Optional[bool]=True, threadsafe: Optional[bool]=True, debug_enabled: Optional[bool]=False) -> ProxyObjT:
        """
        Proxy Object

        args:
            obj_cls: the class of the object
            obj_getter: the function to get the object
            debug_enabled: if True, will raise an error if the object is not found
        """
        assert obj_cls or obj_getter, 'Either `obj_cls` or `obj_getter` must be provided'
        self.__dict__['__obj_cls_'] = obj_cls
        if obj_getter and isinstance(obj_getter, str):
            from lazyops.utils.helpers import lazy_import
            obj_getter = lazy_import(obj_getter)
        self.__dict__['__obj_getter_'] = obj_getter
        self.__dict__['__threadlock_'] = None if threadsafe else threading.Lock()
        self.__dict__['__obj_'] = None
        self.__dict__['__obj_args_'] = obj_args or []
        self.__dict__['__obj_kwargs_'] = obj_kwargs or {}
        self.__dict__['__obj_initialize_'] = obj_initialize
        self.__dict__['__debug_enabled_'] = debug_enabled
        self.__dict__['__last_attrs_'] = {}

    @contextlib.contextmanager
    def _objlock_(self):
        """
        Returns the object lock
        """
        if self.__dict__['__threadlock_'] is not None:
            try:
                with self.__dict__['__threadlock_']:
                    yield
            except Exception as e:
                raise e
        else:
            yield

    @property
    def _obj_(self) -> ProxyObjT:
        """
        Returns the object
        """
        if self.__dict__['__obj_'] is None:
            with self._objlock_():
                if self.__dict__['__obj_getter_']:
                    self.__dict__['__obj_'] = self.__dict__['__obj_getter_'](*self.__dict__['__obj_args_'], **self.__dict__['__obj_kwargs_'])
                elif self.__dict__['__obj_cls_']:
                    if isinstance(self.__dict__['__obj_cls_'], str):
                        from lazyops.utils.helpers import lazy_import
                        self.__dict__['__obj_cls_'] = lazy_import(self.__dict__['__obj_cls_'])
                    if self.__dict__['__obj_initialize_']:
                        self.__dict__['__obj_'] = self.__dict__['__obj_cls_'](*self.__dict__['__obj_args_'], **self.__dict__['__obj_kwargs_'])
                    else:
                        self.__dict__['__obj_'] = self.__dict__['__obj_cls_']
        return self.__dict__['__obj_']

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the proxy object
        """
        return self._obj_(*args, **kwargs)

    def __getattr__(self, name) -> Any:
        """
        Forward all unknown attributes to the proxy object
        """
        if name in self.__dict__:
            return self.__dict__[name]
        if not self.__dict__['__debug_enabled_']:
            return getattr(self._obj_, name)
        if name not in self.__dict__['__last_attrs_']:
            self.__dict__['__last_attrs_'][name] = 0
        self.__dict__['__last_attrs_'][name] += 1
        if self.__dict__['__last_attrs_'][name] > 5:
            raise AttributeError(f'Proxy object has no attribute {name}')
        if hasattr(self._obj_, name):
            self.__dict__['__last_attrs_'][name] = 0
            return getattr(self._obj_, name)
        raise AttributeError(f'Proxy object has no attribute {name}')
    if TYPE_CHECKING:

        def __new__(cls, *args, **kwargs) -> Type[ProxyObjT]:
            ...