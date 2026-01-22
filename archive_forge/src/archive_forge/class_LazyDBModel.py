from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
class LazyDBModel:

    def __init__(self, name, schema, hash_schema: Dict[str, str]=None, is_dev: bool=False):
        self.name = name
        self.base_schema = schema
        self.hash_schema = hash_schema
        self.is_prod = not is_dev
        if self.is_prod:
            logger.info(f'[{self.index_name} Index] Production Mode Enabled. Removing Non-Hashed Values')
        self.create_schema()
        self.index = {}
        self.lookup = {}
        self.idx = 0

    def create_schema(self):
        if isinstance(self.base_schema, str):
            self.base_schema = sjson_loads(self.base_schema)
        assert isinstance(self.base_schema, dict), 'Schema must be a Dict or String that is JSON Decodable'
        self.schema = create_model(self.name, **self.id_schema, **self.base_schema)
        self.schema_props = list(self.schema.schema()['properties'].keys())
        setattr(sys.modules[self.schema.__module__], self.schema.__name__, self.schema)

    @timed_cache(15, 512)
    def validate_idx(self, uid: int=None, dbid: str=None, *args, **kwargs):
        if uid is not None and self.index.get(uid):
            return uid
        if dbid is not None and self.lookup.get(dbid):
            return self.lookup[dbid]
        return None

    async def async_validate_idx(self, uid: int=None, dbid: str=None, *args, **kwargs):
        if uid is not None and self.index.get(uid):
            return uid
        if dbid is not None and self.lookup.get(dbid):
            return self.lookup[dbid]
        return None

    @timed_cache(5)
    def get_by_id(self, uid: int=None, dbid: str=None, *args, **kwargs):
        idx = self.validate_idx(*args, uid=uid, dbid=dbid, **kwargs)
        if idx is None:
            return None
        return self.index.get(idx)

    @timed_cache(15)
    def match_props(self, item, name, val, *args, **kwargs):
        item_data = jsonable_encoder(item)
        if not item_data.get(name):
            return False
        by_type = kwargs.get('ByType', False)
        if by_type:
            return isinstance(item_data[name], val)
        if isinstance(val, list) and isinstance(item_data[name], str):
            return bool(item_data[name] in val)
        if isinstance(val, str) and isinstance(item_data[name], list):
            return bool(val in item_data[name])
        if isinstance(val, str) and val == 'NotNone':
            return bool(item_data[name] is not None)
        if isinstance(val, str) and val == 'NotNoneType':
            return isinstance(item_data[name], NoneType)
        return item_data[name] == val

    @timed_cache(5)
    def get_by_props(self, props: Dict[str, Any], forward=True, *args, **kwargs):
        if not props:
            return None
        search_range = range(self.idx) if forward else range(self.idx, 0, -1)
        for idx in search_range:
            if self.index.get(idx):
                item = self.index[idx]
                for name, val in props.items():
                    if name not in self.schema_props:
                        continue
                    if self.match_props(item, name, val, *args, **kwargs):
                        return item
        return None

    @timed_cache(5)
    def get(self, uid: int=None, dbid: str=None, props: Dict[str, Any]=None, *args, **kwargs):
        logger.info(f'[{self.index_name} Index]: GET Request for UID: {uid}: DBID: {dbid}, Props: {props}, args: {args}, kwargs: {kwargs}')
        return self.get_by_props(*args, props=props, **kwargs) if props is not None else self.get_by_id(*args, uid=uid, dbid=dbid, **kwargs)

    async def async_get(self, uid: int=None, dbid: str=None, props: Dict[str, Any]=None, *args, **kwargs):
        return self.get_by_props(*args, props=props, **kwargs) if props else self.get_by_id(*args, uid=uid, dbid=dbid, **kwargs)

    def filter_items(self, item_list: List[Any], props: Dict[str, Any]=None, *args, **kwargs):
        res_list = []
        for item in item_list:
            for name, val in props.items():
                if name not in self.schema_props:
                    continue
                if self.match_props(item, name, val, *args, **kwargs):
                    res_list.append(item)
        return res_list

    def get_many(self, id_list: List[int]=None, dbid_list: List[int]=None, props: Dict[str, Any]=None, *args, **kwargs):
        if not id_list and (not dbid_list):
            return None
        get_res = [self.get(uid=idx) for idx in id_list] if id_list else [self.get(dbid=dbid) for dbid in dbid_list]
        get_res = [i for i in get_res if i]
        if props:
            get_res = self.filter_items(get_res, props, *args, **kwargs)
        return get_res

    async def async_get_many(self, id_list: List[int]=None, dbid_list: List[int]=None, props: Dict[str, Any]=None, *args, **kwargs):
        if not id_list and (not dbid_list):
            return None
        get_list = id_list or dbid_list
        tasks = [asyncio.ensure_future(self.async_get(uid=idx)) for idx in get_list]
        all_tasks = await asyncio.gather(*tasks)
        get_res = [i for i in all_tasks if i]
        if props:
            get_res = self.filter_items(get_res, props, *args, **kwargs)
        return get_res

    def create(self, data, *args, **kwargs):
        data = self.create_or_update_hash(data)
        new_item = self.schema(*args, uid=self.get_idx(), dbid=self.get_dbid(), **data, **kwargs)
        self.index[self.idx] = new_item
        self.lookup[self.current_id] = self.idx
        self.idx += 1
        return new_item

    async def async_create(self, data, *args, **kwargs):
        data = self.create_or_update_hash(data)
        new_item = self.schema(*args, uid=self.get_idx(), dbid=self.get_dbid(), **data, **kwargs)
        self.index[self.idx] = new_item
        self.lookup[self.current_id] = self.idx
        self.idx += 1
        return new_item

    def remove(self, uid: int=None, dbid: str=None, *args, **kwargs):
        rm_id = self.validate_idx(*args, uid=uid, dbid=dbid, **kwargs)
        if not rm_id:
            return None
        item = self.index.pop(rm_id, None)
        _ = self.lookup.pop(item.dbid)
        return rm_id

    async def async_remove(self, uid: int=None, dbid: str=None, *args, **kwargs):
        rm_id = await self.async_validate_idx(*args, uid=uid, dbid=dbid, **kwargs)
        if not rm_id:
            return None
        item = self.index.pop(rm_id, None)
        _ = self.lookup.pop(item.dbid)
        return rm_id

    def update(self, data: Union[UpdateSchemaType, Dict[str, Any]], uid: int=None, dbid: str=None, prop_name: str=None, prop_val: Any=None, *args, **kwargs):
        item = self.get(id, dbid, *args, prop_name=prop_name, prop_val=prop_val, **kwargs)
        if not item:
            return None
        item_data = jsonable_encoder(item)
        update_data = data if isinstance(data, dict) else data.dict(exclude_unset=True)
        item, update_data = self.create_or_update_hash(update_data=update_data, item=item)
        for field in item_data:
            if field in update_data:
                setattr(item, field, update_data[field])
        idx = item.uid
        item.updated = self.get_timestamp()
        self.index[idx] = item
        return self.index[idx]

    async def async_update(self, data: Union[UpdateSchemaType, Dict[str, Any]], uid: int=None, dbid: str=None, prop_name: str=None, prop_val: Any=None, *args, **kwargs):
        item = await self.async_get(uid, dbid, *args, prop_name=prop_name, prop_val=prop_val, **kwargs)
        if not item:
            return None
        item_data = jsonable_encoder(item)
        update_data = data if isinstance(data, dict) else data.dict(exclude_unset=True)
        item, update_data = self.create_or_update_hash(update_data=update_data, item=item)
        for field in item_data:
            if field in update_data:
                setattr(item, field, update_data[field])
        idx = item.uid
        item.updated = self.get_timestamp()
        self.index[idx] = item
        return self.index[idx]

    def clear_caches(self):
        raise NotImplementedError

    @cached_property
    def model_name(self):
        return self.name.capitalize()

    @cached_property
    def logger_name(self):
        return f'[{self.model_name} Model]'

    @property
    def id_schema(self):
        return {'uid': (int, Field(default_factory=self.get_idx)), 'dbid': (str, Field(default_factory=self.get_dbid)), 'created': (str, Field(default_factory=self.get_timestamp)), 'updated': (str, Field(default_factory=self.get_timestamp))}

    @property
    def current_id(self):
        return f'{self.name}_{self.idx}'

    @property
    def has_hashing(self):
        return bool(self.hash_schema)

    def get_idx(self):
        return self.idx

    def get_dbid(self):
        return self.current_id

    def get_timestamp(self):
        return tstamp()

    def create_or_update_hash(self, data=None, update_data=None, item=None):
        if not self.has_hashing:
            if data:
                return data
            return (item, update_data)
        logger.info(self.hash_schema)
        for field, hash_field in self.hash_schema.items():
            if data and data.get(field):
                data[hash_field] = self.create_hash(val=data[field])
                if self.is_prod:
                    _ = data.pop(field)
            elif update_data and update_data.get(field):
                item, updated = self.update_hash(prop=field, hash_prop=hash_field, new_val=update_data[field], do_verify=True, item=item)
                if updated:
                    logger.info(f'[{self.index_name} Index]: Updated Hashed Field: {field} = {hash_field}')
                    if not self.is_prod:
                        setattr(item, field, update_data[field])
                else:
                    logger.error(f'[{self.index_name} Index]: Failed Validation for Hash Field {field} = {hash_field}')
                _ = update_data.pop(field)
        if data:
            return data
        return (item, update_data)

    @classmethod
    def create_hash(cls, prop=None, hash_prop=None, item=None, item_data=None, val=None, *args, **kw):
        if item is None and item_data is None and (val is None):
            raise ValueError
        if val:
            return LazyHasher.create(val)
        item_data = item_data or jsonable_encoder(item)
        hash_results = LazyHasher.create(item_data[prop])
        if not item:
            return {hash_prop: hash_results}
        setattr(item, hash_prop, LazyHasher.create[item_data[prop]])
        return item

    @classmethod
    def update_hash(cls, prop, hash_prop, new_val, do_verify=True, item=None, item_data=None, *args, **kw):
        if item is None and item_data is None:
            raise ValueError
        item_data = item_data or jsonable_encoder(item)
        hash_results = LazyHasher.update(item_data[hash_prop], item_data[prop], new_val, do_verify=do_verify)
        if not item:
            return {'updated': bool(hash_results), hash_prop: hash_results}
        if not hash_results:
            return (item, False)
        setattr(item, hash_prop, hash_results)
        return (item, True)

    def __call__(self, method, uid: int=None, dbid: str=None, *args, **kwargs):
        func = getattr(self, method)
        return func(uid=uid, dbid=dbid, **kwargs)