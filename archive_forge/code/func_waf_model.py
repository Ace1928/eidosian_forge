import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def waf_model(name):
    waf_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(waf_data))
    return waf_models.get_waiter(name)