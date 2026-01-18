import numpy as np
import pytest
from unittest.mock import Mock, patch
@patch('numpy.__config__._check_pyyaml')
def test_pyyaml_not_found(self, mock_yaml_importer):
    mock_yaml_importer.side_effect = ModuleNotFoundError()
    with pytest.warns(UserWarning):
        np.show_config()