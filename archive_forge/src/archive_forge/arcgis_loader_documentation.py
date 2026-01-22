from __future__ import annotations
import json
import re
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazy load records from FeatureLayer.