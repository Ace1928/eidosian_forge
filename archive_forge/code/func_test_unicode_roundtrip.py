import re
import pytest
def test_unicode_roundtrip(protocol_real):
    shell_id = protocol_real.open_shell(codepage=65001)
    command_id = protocol_real.run_command(shell_id, u'PowerShell', arguments=['-Command', 'Write-Host', u'こんにちは'])
    try:
        std_out, std_err, status_code = protocol_real.get_command_output(shell_id, command_id)
        assert status_code == 0
        assert len(std_err) == 0
        assert std_out == u'こんにちは\n'.encode('utf-8')
    finally:
        protocol_real.cleanup_command(shell_id, command_id)
        protocol_real.close_shell(shell_id)