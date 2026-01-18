from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
def test_transformer_stats_logger_raises():
    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerLogger()
        logger.log('test log')
    with pytest.raises(ValueError, match='No active transformer'):
        logger = cirq.TransformerLogger()
        logger.register_initial(cirq.Circuit(), 'stage-1')
        logger.register_final(cirq.Circuit(), 'stage-1')
        logger.log('test log')
    with pytest.raises(ValueError, match='currently active transformer stage-2'):
        logger = cirq.TransformerLogger()
        logger.register_initial(cirq.Circuit(), 'stage-2')
        logger.register_final(cirq.Circuit(), 'stage-3')