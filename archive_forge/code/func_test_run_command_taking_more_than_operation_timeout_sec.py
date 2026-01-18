import re
import pytest
def test_run_command_taking_more_than_operation_timeout_sec(protocol_real):
    shell_id = protocol_real.open_shell()
    command_id = protocol_real.run_command(shell_id, 'PowerShell -Command Start-Sleep -s {0}'.format(protocol_real.operation_timeout_sec * 2))
    assert re.match('^\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}$', command_id)
    std_out, std_err, status_code = protocol_real.get_command_output(shell_id, command_id)
    assert status_code == 0
    assert len(std_err) == 0
    protocol_real.cleanup_command(shell_id, command_id)
    protocol_real.close_shell(shell_id)