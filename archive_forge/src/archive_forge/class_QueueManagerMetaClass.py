from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
class QueueManagerMetaClass(type):
    """
    Global Queue Manager
    """
    name: Optional[str] = 'queuemanager'
    _queues: Dict[str, 'TaskQueue'] = {}
    _queue_schemas: Dict[str, Dict[str, Dict[str, Any]]] = None
    _settings: Optional['BaseSettings'] = None
    _module_name: str = None

    @property
    def module_name(cls) -> str:
        """
        Returns the module name
        """
        if cls._module_name is None:
            cls._module_name = cls.__module__.split('.', 1)[0].strip()
        return cls._module_name

    def get_settings(cls) -> 'BaseSettings':
        """
        Returns the settings object

        - Should be overwritten by the subclass
        """
        raise NotImplementedError

    @property
    def settings(cls) -> 'BaseSettings':
        """
        Returns the settings object
        """
        if cls._settings is None:
            cls._settings = cls.get_settings()
            register_client(cls, name=f'{cls.module_name}.{cls.name}')
        return cls._settings

    def get_queue_schemas(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns the queue schemas

        - Should be overwritten by the subclass
        """
        raise NotImplementedError

    @property
    def queue_schemas(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns the queue schemas
        """
        if cls._queue_schemas is None:
            cls._queue_schemas = cls.get_queue_schemas()
        return cls._queue_schemas

    def get_task_queue(cls, name: str, kind: Optional[str]=None) -> 'TaskQueue':
        """
        Returns a Task Queue

        - Should be overwritten by the subclass
        """
        raise NotImplementedError

    def get_or_init_queue(cls, queue_name: str, kind: Optional[str]=None) -> 'TaskQueue':
        """
        Returns the queue object
        """
        if queue_name not in cls._queues:
            if kind is None:
                for k, v in cls.queue_schemas.items():
                    if queue_name in v:
                        kind = k
                        break
            try:
                q = cls.get_task_queue(name=queue_name, kind=kind)
            except Exception as e:
                q = False
            cls._queues[queue_name] = q
        return cls._queues[queue_name]

    def get_queue(cls, queue_name: str, kind: Optional[str]=None) -> Optional['TaskQueue']:
        """
        Returns the queue object
        """
        return cls.get_or_init_queue(queue_name=queue_name, kind=kind)

    def __getitem__(cls, queue_name: str) -> Optional['TaskQueue']:
        """
        Returns the queue object
        """
        return cls.get_queue(queue_name=queue_name)