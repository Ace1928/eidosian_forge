from __future__ import annotations
from datetime import datetime
from typing import Iterator, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
Lazily load weather data for the given locations.