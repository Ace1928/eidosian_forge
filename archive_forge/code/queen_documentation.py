from __future__ import annotations
import json
import logging
import os
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
Load assimilated data from a file.