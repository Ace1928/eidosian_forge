import inspect
import pickle
from functools import wraps
from pathlib import Path
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def wandb_log(func=None, datasets=False, models=False, others=False, settings=None):
    """Automatically log parameters and artifacts to W&B by type dispatch.

    This decorator can be applied to a flow, step, or both.
    - Decorating a step will enable or disable logging for certain types within that step
    - Decorating the flow is equivalent to decorating all steps with a default
    - Decorating a step after decorating the flow will overwrite the flow decoration

    Arguments:
        func: (`Callable`). The method or class being decorated (if decorating a step or flow respectively).
        datasets: (`bool`). If `True`, log datasets.  Datasets can be a `pd.DataFrame` or `pathlib.Path`.  The default value is `False`, so datasets are not logged.
        models: (`bool`). If `True`, log models.  Models can be a `nn.Module` or `sklearn.base.BaseEstimator`.  The default value is `False`, so models are not logged.
        others: (`bool`). If `True`, log anything pickle-able.  The default value is `False`, so files are not logged.
        settings: (`wandb.sdk.wandb_settings.Settings`). Custom settings passed to `wandb.init`.  The default value is `None`, and is the same as passing `wandb.Settings()`.  If `settings.run_group` is `None`, it will be set to `{flow_name}/{run_id}.  If `settings.run_job_type` is `None`, it will be set to `{run_job_type}/{step_name}`
    """

    @wraps(func)
    def decorator(func):
        if inspect.isclass(func):
            cls = func
            for attr in cls.__dict__:
                if callable(getattr(cls, attr)):
                    if not hasattr(attr, '_base_func'):
                        setattr(cls, attr, decorator(getattr(cls, attr)))
            return cls
        if hasattr(func, '_base_func'):
            return func

        @wraps(func)
        def wrapper(self, *args, settings=settings, **kwargs):
            if not isinstance(settings, wandb.sdk.wandb_settings.Settings):
                settings = wandb.Settings()
            settings.update(run_group=coalesce(settings.run_group, f'{current.flow_name}/{current.run_id}'), source=wandb.sdk.wandb_settings.Source.INIT)
            settings.update(run_job_type=coalesce(settings.run_job_type, current.step_name), source=wandb.sdk.wandb_settings.Source.INIT)
            with wandb.init(settings=settings) as run:
                with wb_telemetry.context(run=run) as tel:
                    tel.feature.metaflow = True
                proxy = ArtifactProxy(self)
                run.config.update(proxy.params)
                func(proxy, *args, **kwargs)
                for name, data in proxy.inputs.items():
                    wandb_use(name, data, datasets=datasets, models=models, others=others, run=run)
                for name, data in proxy.outputs.items():
                    wandb_track(name, data, datasets=datasets, models=models, others=others, run=run)
        wrapper._base_func = func
        wrapper._kwargs = {'datasets': datasets, 'models': models, 'others': others, 'settings': settings}
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)