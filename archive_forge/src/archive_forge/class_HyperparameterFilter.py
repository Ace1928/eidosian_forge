from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@dataclasses.dataclass(frozen=True)
class HyperparameterFilter:
    """A constraint based on hyperparameter value.

    Attributes:
      hyperparameter_name: A string identifier for the hyperparameter to use for
        the filter. It corresponds to the hyperparameter_name field in the
        Hyperparameter class.
      filter_type: A HyperparameterFilterType describing how we represent the
        filter values in the 'filter' attribute.
      filter: A representation of the set of the filter values.

        If filter_type is REGEX, a str containing the regular expression.
        If filter_type is INTERVAL, a Tuple[float, float] describing the min and
          max values of the filter interval.
        If filter_type is DISCRETE a Collection[float|str|bool] describing the
          finite set of filter values.
    """
    hyperparameter_name: str
    filter_type: HyperparameterFilterType
    filter: Union[str, Tuple[float, float], Collection[Union[float, str, bool]]]