from breezy.tests.per_repository import (TestCaseWithRepository,
def all_repository_vf_format_scenarios():
    scenarios = []
    for test_name, scenario_info in all_repository_format_scenarios():
        format = scenario_info['repository_format']
        if format.supports_full_versioned_files:
            scenarios.append((test_name, scenario_info))
    return scenarios