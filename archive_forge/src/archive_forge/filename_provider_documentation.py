from typing import Any, Dict, Optional
from ray.data.block import Block
from ray.util.annotations import PublicAPI
Generate a filename for a row.

        .. note::
            Filenames must be unique and deterministic for a given task, block, and row
            index.

            A block consists of multiple rows, and each row corresponds to a single
            output file. Each task might produce a different number of blocks, and each
            block might contain a different number of rows.

        .. tip::
            If you require a contiguous row index into the global dataset, use
            :meth:`~Dataset.iter_rows`. This method is single-threaded and isn't
            recommended for large datasets.

        Args:
            row: The row that will be written to a file.
            task_index: The index of the the write task.
            block_index: The index of the block *within* the write task.
            row_index: The index of the row *within* the block.
        