from __future__ import annotations
import logging # isort:skip
import csv
from typing import TypedDict
from ..util.sampledata import external_path, open_csv
class StockData(TypedDict):
    date: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[int]
    adj_close: list[float]