from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
CSV datasource, for reading and writing CSV files.