import pytest
from winrm.protocol import Protocol
def test_run_command_with_arguments_and_cleanup_command(protocol_fake):
    shell_id = protocol_fake.open_shell()
    command_id = protocol_fake.run_command(shell_id, 'ipconfig', ['/all'])
    assert command_id == '11111111-1111-1111-1111-111111111114'
    protocol_fake.cleanup_command(shell_id, command_id)
    protocol_fake.close_shell(shell_id)