from hypothesis import HealthCheck, settings
def _activateHypothesisProfile() -> None:
    """
    Load a Hypothesis profile appropriate for a Twisted test suite.
    """
    deterministic = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], derandomize=True)
    settings.register_profile('twisted_trial_test_profile_deterministic', deterministic)
    settings.load_profile('twisted_trial_test_profile_deterministic')