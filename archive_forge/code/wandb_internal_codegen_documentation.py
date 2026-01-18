import os
import pathlib
import grpc_tools  # type: ignore
from grpc_tools import protoc  # type: ignore
import importlib.metadata
from packaging import version

    Generate a class definition listing the deprecated features.
    This is to allow static checks to ensure that proper field names are used.
    