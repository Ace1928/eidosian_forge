from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
def test_transformer_stats_logger_show_levels(capfd):
    q = cirq.LineQubit.range(2)
    context = cirq.TransformerContext(logger=cirq.TransformerLogger())
    initial_circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.CNOT(*q))
    _ = t1(initial_circuit, context=context)
    info_line = 'LogLevel.INFO Second INFO Log of T1'
    debug_line = 'LogLevel.DEBUG First Verbose Log of T1'
    warning_line = 'LogLevel.WARNING Third WARNING Log of T1'
    for level in [LogLevel.ALL, LogLevel.DEBUG]:
        context.logger.show(level)
        out, _ = capfd.readouterr()
        assert all((line in out for line in [info_line, debug_line, warning_line]))
    context.logger.show(LogLevel.INFO)
    out, _ = capfd.readouterr()
    assert info_line in out and warning_line in out and (debug_line not in out)
    context.logger.show(LogLevel.DEBUG)
    out, _ = capfd.readouterr()
    assert info_line in out and warning_line in out and (debug_line in out)
    context.logger.show(LogLevel.NONE)
    out, _ = capfd.readouterr()
    assert all((line not in out for line in [info_line, debug_line, warning_line]))