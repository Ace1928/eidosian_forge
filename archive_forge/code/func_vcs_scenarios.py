from breezy import foreign, tests
def vcs_scenarios():
    scenarios = []
    for name, vcs in foreign.foreign_vcs_registry.items():
        scenarios.append((vcs.__class__.__name__, {'branch_factory': vcs.branch_format.get_foreign_tests_branch_factory(), 'repository_factory': vcs.repository_format.get_foreign_tests_repository_factory(), 'branch_format': vcs.branch_format, 'repository_format': vcs.repository_format}))
    return scenarios