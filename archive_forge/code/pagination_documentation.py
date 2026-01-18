from typing import Any, List, Generic, TypeVar, Optional, cast
from typing_extensions import Protocol, override, runtime_checkable
from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage

        This page represents a response that isn't actually paginated at the API level
        so there will never be a next page.
        