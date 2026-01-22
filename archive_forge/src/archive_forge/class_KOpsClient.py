from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
class KOpsClient(BaseKOpsClient):

    @classmethod
    def get_namespaces(cls, **kwargs) -> List[str]:
        return get_namespaces(**kwargs)

    @classmethod
    async def aget_namespaces(cls, **kwargs) -> List[str]:
        return await aget_namespaces(**kwargs)

    @classmethod
    def get_nodes(cls, **kwargs) -> List[t.V1Node]:
        return get_nodes(**kwargs)

    @classmethod
    async def aget_nodes(cls, **kwargs) -> List[at.V1Node]:
        return await aget_nodes(**kwargs)

    @classmethod
    def get_pods(cls, namespace: Optional[str]=None, **kwargs) -> List[t.V1Pod]:
        return get_pods(namespace=namespace, **kwargs)

    @classmethod
    async def aget_pods(cls, namespace: Optional[str]=None, **kwargs) -> List[at.V1Pod]:
        return await aget_pods(namespace=namespace, **kwargs)

    @classmethod
    def run_command(cls, name: str, namespace: str, command: Union[str, List[str]], container: Optional[str]=None, stderr: Optional[bool]=True, stdin: Optional[bool]=True, stdout: Optional[bool]=True, tty: Optional[bool]=False, ignore_error: bool=True, **kwargs) -> str:
        return run_pod_command(name=name, namespace=namespace, command=command, container=container, stderr=stderr, stdin=stdin, stdout=stdout, tty=tty, ignore_error=ignore_error, **kwargs)

    @classmethod
    async def arun_command(cls, name: str, namespace: str, command: Union[str, List[str]], container: Optional[str]=None, stderr: Optional[bool]=True, stdin: Optional[bool]=True, stdout: Optional[bool]=True, tty: Optional[bool]=False, ignore_error: bool=True, **kwargs) -> str:
        return await arun_pod_command(name=name, namespace=namespace, command=command, container=container, stderr=stderr, stdin=stdin, stdout=stdout, tty=tty, ignore_error=ignore_error, **kwargs)