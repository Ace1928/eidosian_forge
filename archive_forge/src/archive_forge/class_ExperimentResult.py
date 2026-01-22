import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pyquil.experiment._setting import ExperimentSetting
@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting
    in a tomographic experiment.

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """
    setting: ExperimentSetting
    expectation: Union[float, complex]
    total_counts: int
    std_err: Optional[Union[float, complex]] = None
    raw_expectation: Optional[Union[float, complex]] = None
    raw_std_err: Optional[float] = None
    calibration_expectation: Optional[Union[float, complex]] = None
    calibration_std_err: Optional[Union[float, complex]] = None
    calibration_counts: Optional[int] = None
    additional_results: Optional[List['ExperimentResult']] = None

    def __init__(self, setting: ExperimentSetting, expectation: Union[float, complex], total_counts: int, std_err: Optional[Union[float, complex]]=None, raw_expectation: Optional[Union[float, complex]]=None, raw_std_err: Optional[Union[float, complex]]=None, calibration_expectation: Optional[Union[float, complex]]=None, calibration_std_err: Optional[Union[float, complex]]=None, calibration_counts: Optional[int]=None, additional_results: Optional[List['ExperimentResult']]=None):
        object.__setattr__(self, 'setting', setting)
        object.__setattr__(self, 'expectation', expectation)
        object.__setattr__(self, 'total_counts', total_counts)
        object.__setattr__(self, 'raw_expectation', raw_expectation)
        object.__setattr__(self, 'calibration_expectation', calibration_expectation)
        object.__setattr__(self, 'calibration_counts', calibration_counts)
        object.__setattr__(self, 'additional_results', additional_results)
        object.__setattr__(self, 'std_err', std_err)
        object.__setattr__(self, 'raw_std_err', raw_std_err)
        object.__setattr__(self, 'calibration_std_err', calibration_std_err)

    def __str__(self) -> str:
        return f'{self.setting}: {self.expectation} +- {self.std_err}'

    def __repr__(self) -> str:
        return f'ExperimentResult[{self}]'

    def serializable(self) -> Dict[str, Any]:
        return {'type': 'ExperimentResult', 'setting': self.setting, 'expectation': self.expectation, 'std_err': self.std_err, 'total_counts': self.total_counts, 'raw_expectation': self.raw_expectation, 'raw_std_err': self.raw_std_err, 'calibration_expectation': self.calibration_expectation, 'calibration_std_err': self.calibration_std_err, 'calibration_counts': self.calibration_counts}