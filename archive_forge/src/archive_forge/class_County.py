from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class County:
    name = 'San Sebastián'
    state = 'PR'

    def __repr__(self) -> str:
        return self.name + ', ' + self.state