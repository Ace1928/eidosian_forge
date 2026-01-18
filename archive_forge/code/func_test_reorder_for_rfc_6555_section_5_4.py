from __future__ import annotations
import socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Sequence
import attrs
import pytest
import trio
from trio._highlevel_open_tcp_stream import (
from trio.socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, SocketType
from trio.testing import Matcher, RaisesGroup
def test_reorder_for_rfc_6555_section_5_4() -> None:

    def fake4(i: int) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        return (AF_INET, SOCK_STREAM, IPPROTO_TCP, '', (f'10.0.0.{i}', 80))

    def fake6(i: int) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        return (AF_INET6, SOCK_STREAM, IPPROTO_TCP, '', (f'::{i}', 80))
    for fake in (fake4, fake6):
        targets = [fake(0), fake(1), fake(2)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0), fake(1), fake(2)]
        targets = [fake(0)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0)]
    orig = [fake4(0), fake6(0), fake4(1), fake6(1)]
    targets = list(orig)
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == orig
    targets = [fake4(0), fake4(1), fake4(2), fake6(0), fake6(1)]
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == [fake4(0), fake6(0), fake4(1), fake4(2), fake6(1)]