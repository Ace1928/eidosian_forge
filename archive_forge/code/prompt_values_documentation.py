from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Sequence, cast
from typing_extensions import TypedDict
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
Get the namespace of the langchain object.