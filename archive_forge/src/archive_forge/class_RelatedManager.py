import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from django.db import (
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData, resolve_callables
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
class RelatedManager(superclass, AltersData):

    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.model = rel.related_model
        self.field = rel.field
        self.core_filters = {self.field.name: instance}

    def __call__(self, *, manager):
        manager = getattr(self.model, manager)
        manager_class = create_reverse_many_to_one_manager(manager.__class__, rel)
        return manager_class(self.instance)
    do_not_call_in_templates = True

    def _check_fk_val(self):
        for field in self.field.foreign_related_fields:
            if getattr(self.instance, field.attname) is None:
                raise ValueError(f'"{self.instance!r}" needs to have a value for field "{field.attname}" before this relationship can be used.')

    def _apply_rel_filters(self, queryset):
        """
            Filter the queryset for the instance this manager is bound to.
            """
        db = self._db or router.db_for_read(self.model, instance=self.instance)
        empty_strings_as_null = connections[db].features.interprets_empty_strings_as_nulls
        queryset._add_hints(instance=self.instance)
        if self._db:
            queryset = queryset.using(self._db)
        queryset._defer_next_filter = True
        queryset = queryset.filter(**self.core_filters)
        for field in self.field.foreign_related_fields:
            val = getattr(self.instance, field.attname)
            if val is None or (val == '' and empty_strings_as_null):
                return queryset.none()
        if self.field.many_to_one:
            try:
                target_field = self.field.target_field
            except FieldError:
                rel_obj_id = tuple([getattr(self.instance, target_field.attname) for target_field in self.field.path_infos[-1].target_fields])
            else:
                rel_obj_id = getattr(self.instance, target_field.attname)
            queryset._known_related_objects = {self.field: {rel_obj_id: self.instance}}
        return queryset

    def _remove_prefetched_objects(self):
        try:
            self.instance._prefetched_objects_cache.pop(self.field.remote_field.get_cache_name())
        except (AttributeError, KeyError):
            pass

    def get_queryset(self):
        if self.instance.pk is None:
            raise ValueError(f'{self.instance.__class__.__name__!r} instance needs to have a primary key value before this relationship can be used.')
        try:
            return self.instance._prefetched_objects_cache[self.field.remote_field.get_cache_name()]
        except (AttributeError, KeyError):
            queryset = super().get_queryset()
            return self._apply_rel_filters(queryset)

    def get_prefetch_queryset(self, instances, queryset=None):
        warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
        if queryset is None:
            return self.get_prefetch_querysets(instances)
        return self.get_prefetch_querysets(instances, [queryset])

    def get_prefetch_querysets(self, instances, querysets=None):
        if querysets and len(querysets) != 1:
            raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
        queryset = querysets[0] if querysets else super().get_queryset()
        queryset._add_hints(instance=instances[0])
        queryset = queryset.using(queryset._db or self._db)
        rel_obj_attr = self.field.get_local_related_value
        instance_attr = self.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        queryset = _filter_prefetch_queryset(queryset, self.field.name, instances)
        for rel_obj in queryset:
            if not self.field.is_cached(rel_obj):
                instance = instances_dict[rel_obj_attr(rel_obj)]
                setattr(rel_obj, self.field.name, instance)
        cache_name = self.field.remote_field.get_cache_name()
        return (queryset, rel_obj_attr, instance_attr, False, cache_name, False)

    def add(self, *objs, bulk=True):
        self._check_fk_val()
        self._remove_prefetched_objects()
        db = router.db_for_write(self.model, instance=self.instance)

        def check_and_update_obj(obj):
            if not isinstance(obj, self.model):
                raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
            setattr(obj, self.field.name, self.instance)
        if bulk:
            pks = []
            for obj in objs:
                check_and_update_obj(obj)
                if obj._state.adding or obj._state.db != db:
                    raise ValueError("%r instance isn't saved. Use bulk=False or save the object first." % obj)
                pks.append(obj.pk)
            self.model._base_manager.using(db).filter(pk__in=pks).update(**{self.field.name: self.instance})
        else:
            with transaction.atomic(using=db, savepoint=False):
                for obj in objs:
                    check_and_update_obj(obj)
                    obj.save()
    add.alters_data = True

    async def aadd(self, *objs, bulk=True):
        return await sync_to_async(self.add)(*objs, bulk=bulk)
    aadd.alters_data = True

    def create(self, **kwargs):
        self._check_fk_val()
        kwargs[self.field.name] = self.instance
        db = router.db_for_write(self.model, instance=self.instance)
        return super(RelatedManager, self.db_manager(db)).create(**kwargs)
    create.alters_data = True

    async def acreate(self, **kwargs):
        return await sync_to_async(self.create)(**kwargs)
    acreate.alters_data = True

    def get_or_create(self, **kwargs):
        self._check_fk_val()
        kwargs[self.field.name] = self.instance
        db = router.db_for_write(self.model, instance=self.instance)
        return super(RelatedManager, self.db_manager(db)).get_or_create(**kwargs)
    get_or_create.alters_data = True

    async def aget_or_create(self, **kwargs):
        return await sync_to_async(self.get_or_create)(**kwargs)
    aget_or_create.alters_data = True

    def update_or_create(self, **kwargs):
        self._check_fk_val()
        kwargs[self.field.name] = self.instance
        db = router.db_for_write(self.model, instance=self.instance)
        return super(RelatedManager, self.db_manager(db)).update_or_create(**kwargs)
    update_or_create.alters_data = True

    async def aupdate_or_create(self, **kwargs):
        return await sync_to_async(self.update_or_create)(**kwargs)
    aupdate_or_create.alters_data = True
    if rel.field.null:

        def remove(self, *objs, bulk=True):
            if not objs:
                return
            self._check_fk_val()
            val = self.field.get_foreign_related_value(self.instance)
            old_ids = set()
            for obj in objs:
                if not isinstance(obj, self.model):
                    raise TypeError("'%s' instance expected, got %r" % (self.model._meta.object_name, obj))
                if self.field.get_local_related_value(obj) == val:
                    old_ids.add(obj.pk)
                else:
                    raise self.field.remote_field.model.DoesNotExist('%r is not related to %r.' % (obj, self.instance))
            self._clear(self.filter(pk__in=old_ids), bulk)
        remove.alters_data = True

        async def aremove(self, *objs, bulk=True):
            return await sync_to_async(self.remove)(*objs, bulk=bulk)
        aremove.alters_data = True

        def clear(self, *, bulk=True):
            self._check_fk_val()
            self._clear(self, bulk)
        clear.alters_data = True

        async def aclear(self, *, bulk=True):
            return await sync_to_async(self.clear)(bulk=bulk)
        aclear.alters_data = True

        def _clear(self, queryset, bulk):
            self._remove_prefetched_objects()
            db = router.db_for_write(self.model, instance=self.instance)
            queryset = queryset.using(db)
            if bulk:
                queryset.update(**{self.field.name: None})
            else:
                with transaction.atomic(using=db, savepoint=False):
                    for obj in queryset:
                        setattr(obj, self.field.name, None)
                        obj.save(update_fields=[self.field.name])
        _clear.alters_data = True

    def set(self, objs, *, bulk=True, clear=False):
        self._check_fk_val()
        objs = tuple(objs)
        if self.field.null:
            db = router.db_for_write(self.model, instance=self.instance)
            with transaction.atomic(using=db, savepoint=False):
                if clear:
                    self.clear(bulk=bulk)
                    self.add(*objs, bulk=bulk)
                else:
                    old_objs = set(self.using(db).all())
                    new_objs = []
                    for obj in objs:
                        if obj in old_objs:
                            old_objs.remove(obj)
                        else:
                            new_objs.append(obj)
                    self.remove(*old_objs, bulk=bulk)
                    self.add(*new_objs, bulk=bulk)
        else:
            self.add(*objs, bulk=bulk)
    set.alters_data = True

    async def aset(self, objs, *, bulk=True, clear=False):
        return await sync_to_async(self.set)(objs=objs, bulk=bulk, clear=clear)
    aset.alters_data = True