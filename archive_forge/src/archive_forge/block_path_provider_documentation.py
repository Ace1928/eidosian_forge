import posixpath
from typing import TYPE_CHECKING, Optional
from ray.util.annotations import DeveloperAPI

        Resolves and returns the write path for the given dataset block. When
        implementing this method, care should be taken to ensure that a unique
        path is provided for every dataset block.

        Args:
            base_path: The base path to write the dataset block out to. This is
                expected to be the same for all blocks in the dataset, and may
                point to either a directory or file prefix.
            filesystem: The filesystem implementation that will be used to
                write a file out to the write path returned.
            dataset_uuid: Unique identifier for the dataset that this block
                belongs to.
            block: The block to write.
            task_index: Ordered index of the write task within its parent
                dataset.
            block_index: Ordered index of the block to write within its parent
                write task.
            file_format: File format string for the block that can be used as
                the file extension in the write path returned.

        Returns:
            The dataset block write path.
        