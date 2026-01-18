from __future__ import (absolute_import, division, print_function)
import testinfra.utils.ansible_runner
def test_retrieve_secret_with_summon(host):
    result = host.check_output("summon --yaml 'DB_USERNAME: !var ansible/target-password' bash -c 'printenv DB_USERNAME'", shell=True)
    assert result == 'target_secret_password'