from __future__ import annotations
from .base import *
class AsyncKDBDict(KDBDict):
    """
    Async KDBDict with async magic methods
    """

    async def set(self, field: Union[str, int], value: Any, key: Optional[str]=None, ex: Optional[int]=None) -> int:
        """
        Sets the value for the given key
        """
        _key = self.get_key(key)
        r = await self.kdb.async_hset(_key, field, self.encode(value))
        if self.auto_expire:
            await self.kdb.async_expire(_key, ex or self.expiration)
        if self.is_lookup_index and isinstance(value, str) and isinstance(field, int):
            _lkey = self.get_key(key, lookup=True)
            await self.kdb.async_hset(_lkey, value, field)
        return r

    async def ahset(self, field: Union[str, int], value: Any, key: Optional[str]=None, ex: Optional[int]=None) -> int:
        """
        Sets the value for the given key
        """
        return await self.set(field, value, key, ex)

    async def get(self, field: Union[str, int], default: Optional[DT]=None, key: Optional[str]=None) -> DT:
        """
        Returns the value for the given key
        """
        if self.is_lookup_index and isinstance(field, str):
            _lkey = self.get_key(key, lookup=True)
            value = await self.kdb.async_hget(_lkey, field)
            if value is not None:
                return int(value.decode())
        value = await self.kdb.async_hget(self.get_key(key), field)
        return self.decode(value) if value is not None else default

    async def ahget(self, field: Union[str, int], default: Optional[DT]=None, key: Optional[str]=None) -> DT:
        """
        Returns the value for the given key
        """
        return await self.get(field, default, key)

    async def append(self, field: Union[str, int], value: Any, key: Optional[str]=None, ex: Optional[int]=None) -> int:
        """
        Appends the value to the list
        """
        if not await self.exists(field=field, key=key):
            return await self.ahset(field, [value], key, ex=ex)
        return await self.ahset(field=field, value=await self.ahget(field=field, key=key) + [value], key=key, ex=ex)

    async def exists(self, field: Union[str, int], key: Optional[str]=None) -> bool:
        """
        Returns the length of the given key
        """
        if self.is_lookup_index and isinstance(field, str):
            _lkey = self.get_key(key, lookup=True)
            return await self.kdb.async_hexists(_lkey, field)
        return await self.kdb.async_hexists(self.get_key(key), field)

    async def exists_all(self, *fields: Union[str, int], key: Optional[str]=None) -> bool:
        """
        Returns whether all the given keys exist
        """
        _key = self.get_key(key, lookup=self.is_lookup_index)
        all_keys = [k.decode() for k in await self.kdb.async_hkeys(_key)]
        fields = [str(field) for field in fields]
        return all((field in all_keys for field in fields))

    async def ahexists(self, field: Union[str, int], key: Optional[str]=None) -> bool:
        """
        Returns the length of the given key
        """
        return await self.exists(field, key)

    async def __getitem__(self, key: Union[str, int]) -> Any:
        """
        Returns the value for the given key
        # Works for Async
        """
        if key == 'count':
            return int(await self.kdb.async_get(self.name_count_key, 0, _return_raw_value=True, _serializer=True))
        return await self.ahget(key)

    async def delete(self, field: Union[str, int], key: Optional[str]=None) -> int:
        """
        Deletes the given key
        """
        if self.is_lookup_index and isinstance(field, str):
            _lkey = self.get_key(key, lookup=True)
            await self.kdb.async_hdel(_lkey, field)
        return await self.kdb.async_hdel(self.get_key(key), field)

    async def has(self, key: Union[str, int, List[Union[str, int]]]) -> bool:
        """
        Returns whether the given key is in the database
        """
        if key == 'count':
            return await self.kdb.async_exists(self.name_count_key)
        return await self.exists_all(*key) if isinstance(key, list) else await self.exists(key)

    async def aincr(self, field: Union[str, int], amount: int=1, key: Optional[str]=None) -> int:
        """
        Increments the given key
        """
        return await self.ahincr(field=field, amount=amount, key=key)

    async def adecr(self, field: Union[str, int], amount: int=1, key: Optional[str]=None) -> int:
        """
        Decrements the given key
        """
        return await self.ahdecr(field=field, amount=amount, key=key)

    @property
    async def length(self) -> int:
        """
        Returns the length of the given key
        """
        return await self.ahlen()

    @property
    async def idx(self) -> int:
        """
        Returns the index for the given key
        """
        try:
            return await self.kdb.async_incr(self.name_count_key)
        except Exception as e:
            _idx = int(await self.kdb.async_get(self.name_count_key, 0, _return_raw_value=True, _serializer=True))
            await self.kdb.async_set(self.name_count_key, _idx + 1, ex=self.expiration, _serializer=True)
            return _idx + 1

    async def get_idx(self, key: str) -> int:
        """
        Returns the index for the given key
        """
        return await self.ahget(key) if await self.has(key) else await self.idx

    @property
    async def dkeys(self) -> List[str]:
        """
        Returns the keys for the given key
        """
        return await self.ahkeys()

    @property
    async def dvalues(self) -> List[Any]:
        """
        Returns the values for the given key
        """
        return await self.ahvals()
    '\n    Lookup props\n    '

    @property
    async def lkeys(self) -> List[str]:
        """
        Returns the lookup keys
        """
        if not self.is_lookup_index:
            return []
        keys = await self.kdb.async_hkeys(self.get_key(lookup=True))
        return [key.decode() for key in keys]

    @property
    async def lvalues(self) -> List[int]:
        """
        Returns the lookup values
        """
        if not self.is_lookup_index:
            return []
        return [int(value.decode()) for value in await self.kdb.async_hvals(self.get_key(lookup=True))]

    @property
    async def litems(self) -> Dict[str, int]:
        """
        Returns the lookup items
        """
        if not self.is_lookup_index:
            return {}
        items = await self.kdb.async_hgetall(self.get_key(lookup=True))
        return {key.decode(): int(value.decode()) for key, value in items.items()}

    async def lhas(self, *keys: Union[str, int]) -> bool:
        """
        Checks whether the lookup key exists
        """
        if not self.is_lookup_index:
            return False
        lookup_keys = []
        if any((isinstance(key, str) for key in keys)):
            lookup_keys += await self.lkeys
        if any((isinstance(key, int) for key in keys)):
            lookup_keys += await self.lvalues
        return all((key in lookup_keys for key in keys))

    async def get_keys(self, key: Optional[str]=None, **kwargs) -> List[str]:
        """
        Returns the keys for the given key
        """
        match = self.get_match_key(key)
        keys = []
        async for key in self.kdb.async_scan_iter(match=match, **kwargs):
            keys.append(key.decode())
        return keys

    async def get_items(self, key: Optional[str]=None, **kwargs) -> Dict[str, Any]:
        """
        Returns the keys for the given key
        """
        match = self.get_match_key(key)
        items = {}
        async for key in self.kdb.async_scan_iter(match=match, **kwargs):
            items[key.decode()] = await self.kdb.async_get(key)
        return items