from packaging import version
import wandb.util
from wandb.sdk.lib import deprecate
from langchain.callbacks.tracers import WandbTracer  # noqa: E402, I001
class WandbTracer(WandbTracer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecate.deprecate(field_name=deprecate.Deprecated.langchain_tracer, warning_message='This feature is deprecated and has been moved to `langchain`. Enable tracing by setting LANGCHAIN_WANDB_TRACING=true in your environment. See the documentation at https://python.langchain.com/docs/ecosystem/integrations/agent_with_wandb_tracing for guidance. Replace your current import with `from langchain.callbacks.tracers import WandbTracer`.')