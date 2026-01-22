from warnings import warn
from rich.progress import (
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm
class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit='', unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f'? {self.unit}/s', style='progress.data.speed')
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(speed, ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], self.unit_divisor)
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [''], 1)
        precision = 0 if unit == 1 else 1
        return Text(f'{speed / unit:,.{precision}f} {suffix}{self.unit}/s', style='progress.data.speed')