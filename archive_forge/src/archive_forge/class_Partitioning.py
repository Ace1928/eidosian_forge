import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
@dataclass
class Partitioning:
    """Partition scheme used to describe path-based partitions.

    Path-based partition formats embed all partition keys and values directly in
    their dataset file paths.

    For example, to read a dataset with
    `Hive-style partitions <https://athena.guide/articles/hive-style-partitioning/>`_:

        >>> import ray
        >>> from ray.data.datasource.partitioning import Partitioning
        >>> ds = ray.data.read_csv(
        ...     "s3://anonymous@ray-example-data/iris.csv",
        ...     partitioning=Partitioning("hive"),
        ... )

    Instead, if your files are arranged in a directory structure such as:

    .. code::

        root/dog/dog_0.jpeg
        root/dog/dog_1.jpeg
        ...

        root/cat/cat_0.jpeg
        root/cat/cat_1.jpeg
        ...

    Then you can use directory-based partitioning:

        >>> import ray
        >>> from ray.data.datasource.partitioning import Partitioning
        >>> root = "s3://anonymous@air-example-data/cifar-10/images"
        >>> partitioning = Partitioning("dir", field_names=["class"], base_dir=root)
        >>> ds = ray.data.read_images(root, partitioning=partitioning)
    """
    style: PartitionStyle
    base_dir: Optional[str] = None
    field_names: Optional[List[str]] = None
    filesystem: Optional['pyarrow.fs.FileSystem'] = None

    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = ''
        self._normalized_base_dir = None
        self._resolved_filesystem = None

    @property
    def normalized_base_dir(self) -> str:
        """Returns the base directory normalized for compatibility with a filesystem."""
        if self._normalized_base_dir is None:
            self._normalize_base_dir()
        return self._normalized_base_dir

    @property
    def resolved_filesystem(self) -> 'pyarrow.fs.FileSystem':
        """Returns the filesystem resolved for compatibility with a base directory."""
        if self._resolved_filesystem is None:
            self._normalize_base_dir()
        return self._resolved_filesystem

    def _normalize_base_dir(self):
        """Normalizes the partition base directory for compatibility with the
        given filesystem.

        This should be called once a filesystem has been resolved to ensure that this
        base directory is correctly discovered at the root of all partitioned file
        paths.
        """
        from ray.data.datasource.path_util import _resolve_paths_and_filesystem
        paths, self._resolved_filesystem = _resolve_paths_and_filesystem(self.base_dir, self.filesystem)
        assert len(paths) == 1, f'Expected 1 normalized base directory, but found {len(paths)}'
        normalized_base_dir = paths[0]
        if len(normalized_base_dir) and (not normalized_base_dir.endswith('/')):
            normalized_base_dir += '/'
        self._normalized_base_dir = normalized_base_dir