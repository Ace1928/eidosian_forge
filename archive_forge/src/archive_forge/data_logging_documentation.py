import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
Log a set of predictions.

        Intended usage:

        vl.log_predictions(vl.make_predictions(self.model.predict))

        Args:
            predictions (Sequence | Dict[str, Sequence]): A list of prediction vectors or dictionary
                of lists of prediction vectors
            prediction_col_name (str, optional): the name of the prediction column. Defaults to "output".
            val_ndx_col_name (str, optional): The name of the column linking prediction table
                to the validation ata table. Defaults to "val_row".
            table_name (str, optional): name of the prediction table. Defaults to "validation_predictions".
            commit (bool, optional): determines if commit should be called on the logged data. Defaults to False.
        