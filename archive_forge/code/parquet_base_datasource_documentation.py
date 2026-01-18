import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
Return a human-readable name for this datasource.
        This will be used as the names of the read tasks.
        Note: overrides the base `FileBasedDatasource` method.
        