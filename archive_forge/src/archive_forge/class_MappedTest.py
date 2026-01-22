from __future__ import annotations
from typing import Any
import sqlalchemy as sa
from .base import TestBase
from .sql import TablesTest
from .. import assertions
from .. import config
from .. import schema
from ..entities import BasicEntity
from ..entities import ComparableEntity
from ..util import adict
from ... import orm
from ...orm import DeclarativeBase
from ...orm import events as orm_events
from ...orm import registry
class MappedTest(ORMTest, TablesTest, assertions.AssertsExecutionResults):
    run_setup_classes = 'once'
    run_setup_mappers = 'each'
    classes: Any = None

    @config.fixture(autouse=True, scope='class')
    def _setup_tables_test_class(self):
        cls = self.__class__
        cls._init_class()
        if cls.classes is None:
            cls.classes = adict()
        cls._setup_once_tables()
        cls._setup_once_classes()
        cls._setup_once_mappers()
        cls._setup_once_inserts()
        yield
        cls._teardown_once_class()
        cls._teardown_once_metadata_bind()

    @config.fixture(autouse=True, scope='function')
    def _setup_tables_test_instance(self):
        self._setup_each_tables()
        self._setup_each_classes()
        self._setup_each_mappers()
        self._setup_each_inserts()
        yield
        orm.session.close_all_sessions()
        self._teardown_each_mappers()
        self._teardown_each_classes()
        self._teardown_each_tables()

    @classmethod
    def _teardown_once_class(cls):
        cls.classes.clear()

    @classmethod
    def _setup_once_classes(cls):
        if cls.run_setup_classes == 'once':
            cls._with_register_classes(cls.setup_classes)

    @classmethod
    def _setup_once_mappers(cls):
        if cls.run_setup_mappers == 'once':
            cls.mapper_registry, cls.mapper = cls._generate_registry()
            cls._with_register_classes(cls.setup_mappers)

    def _setup_each_mappers(self):
        if self.run_setup_mappers != 'once':
            self.__class__.mapper_registry, self.__class__.mapper = self._generate_registry()
        if self.run_setup_mappers == 'each':
            self._with_register_classes(self.setup_mappers)

    def _setup_each_classes(self):
        if self.run_setup_classes == 'each':
            self._with_register_classes(self.setup_classes)

    @classmethod
    def _generate_registry(cls):
        decl = registry(metadata=cls._tables_metadata)
        return (decl, decl.map_imperatively)

    @classmethod
    def _with_register_classes(cls, fn):
        """Run a setup method, framing the operation with a Base class
        that will catch new subclasses to be established within
        the "classes" registry.

        """
        cls_registry = cls.classes

        class _Base:

            def __init_subclass__(cls) -> None:
                assert cls_registry is not None
                cls_registry[cls.__name__] = cls
                super().__init_subclass__()

        class Basic(BasicEntity, _Base):
            pass

        class Comparable(ComparableEntity, _Base):
            pass
        cls.Basic = Basic
        cls.Comparable = Comparable
        fn()

    def _teardown_each_mappers(self):
        if self.run_setup_mappers != 'once':
            orm.clear_mappers()

    def _teardown_each_classes(self):
        if self.run_setup_classes != 'once':
            self.classes.clear()

    @classmethod
    def setup_classes(cls):
        pass

    @classmethod
    def setup_mappers(cls):
        pass