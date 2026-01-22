import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI(stability='beta')
class PathPartitionFilter:
    """Partition filter for path-based partition formats.

    Used to explicitly keep or reject files based on a custom filter function that
    takes partition keys and values parsed from the file's path as input.
    """

    @staticmethod
    def of(filter_fn: Callable[[Dict[str, str]], bool], style: PartitionStyle=PartitionStyle.HIVE, base_dir: Optional[str]=None, field_names: Optional[List[str]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None) -> 'PathPartitionFilter':
        """Creates a path-based partition filter using a flattened argument list.

        Args:
            filter_fn: Callback used to filter partitions. Takes a dictionary mapping
                partition keys to values as input. Unpartitioned files are denoted with
                an empty input dictionary. Returns `True` to read a file for that
                partition or `False` to skip it. Partition keys and values are always
                strings read from the filesystem path. For example, this removes all
                unpartitioned files:

                .. code:: python

                    lambda d: True if d else False

                This raises an assertion error for any unpartitioned file found:

                .. code:: python

                    def do_assert(val, msg):
                        assert val, msg

                    lambda d: do_assert(d, "Expected all files to be partitioned!")

                And this only reads files from January, 2022 partitions:

                .. code:: python

                    lambda d: d["month"] == "January" and d["year"] == "2022"

            style: The partition style - may be either HIVE or DIRECTORY.
            base_dir: "/"-delimited base directory to start searching for partitions
                (exclusive). File paths outside of this directory will be considered
                unpartitioned. Specify `None` or an empty string to search for
                partitions in all file path directories.
            field_names: The partition key names. Required for DIRECTORY partitioning.
                Optional for HIVE partitioning. When non-empty, the order and length of
                partition key field names must match the order and length of partition
                directories discovered. Partition key field names are not required to
                exist in the dataset schema.
            filesystem: Filesystem that will be used for partition path file I/O.

        Returns:
            The new path-based partition filter.
        """
        scheme = Partitioning(style, base_dir, field_names, filesystem)
        path_partition_parser = PathPartitionParser(scheme)
        return PathPartitionFilter(path_partition_parser, filter_fn)

    def __init__(self, path_partition_parser: PathPartitionParser, filter_fn: Callable[[Dict[str, str]], bool]):
        """Creates a new path-based partition filter based on a parser.

        Args:
            path_partition_parser: The path-based partition parser.
            filter_fn: Callback used to filter partitions. Takes a dictionary mapping
                partition keys to values as input. Unpartitioned files are denoted with
                an empty input dictionary. Returns `True` to read a file for that
                partition or `False` to skip it. Partition keys and values are always
                strings read from the filesystem path. For example, this removes all
                unpartitioned files:
                ``lambda d: True if d else False``
                This raises an assertion error for any unpartitioned file found:
                ``lambda d: assert d, "Expected all files to be partitioned!"``
                And this only reads files from January, 2022 partitions:
                ``lambda d: d["month"] == "January" and d["year"] == "2022"``
        """
        self._parser = path_partition_parser
        self._filter_fn = filter_fn

    def __call__(self, paths: List[str]) -> List[str]:
        """Returns all paths that pass this partition scheme's partition filter.

        If no partition filter is set, then returns all input paths. If a base
        directory is set, then only paths under this base directory will be parsed
        for partitions. All paths outside of this base directory will automatically
        be considered unpartitioned, and passed into the filter function as empty
        dictionaries.

        Also normalizes the partition base directory for compatibility with the
        given filesystem before applying the filter.

        Args:
            paths: Paths to pass through the partition filter function. All
                paths should be normalized for compatibility with the given
                filesystem.
        Returns:
            List of paths that pass the partition filter, or all paths if no
            partition filter is defined.
        """
        filtered_paths = paths
        if self._filter_fn is not None:
            filtered_paths = [path for path in paths if self._filter_fn(self._parser(path))]
        return filtered_paths

    @property
    def parser(self) -> PathPartitionParser:
        """Returns the path partition parser for this filter."""
        return self._parser