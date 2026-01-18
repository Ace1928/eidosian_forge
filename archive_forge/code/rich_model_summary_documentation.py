from typing import Any, List, Tuple
from typing_extensions import override
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count
Generates a summary of all layers in a :class:`~pytorch_lightning.core.LightningModule` with `rich text
    formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichModelSummary

        trainer = Trainer(callbacks=RichModelSummary())

    You could also enable ``RichModelSummary`` using the :class:`~pytorch_lightning.callbacks.RichProgressBar`

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off.
        **summarize_kwargs: Additional arguments to pass to the `summarize` method.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.

    