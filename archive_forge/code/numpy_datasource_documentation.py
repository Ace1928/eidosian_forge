from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
Numpy datasource, for reading and writing Numpy files.