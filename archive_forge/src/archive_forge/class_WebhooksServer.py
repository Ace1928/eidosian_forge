import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
@experimental
class WebhooksServer:
    """
    The [`WebhooksServer`] class lets you create an instance of a Gradio app that can receive Huggingface webhooks.
    These webhooks can be registered using the [`~WebhooksServer.add_webhook`] decorator. Webhook endpoints are added to
    the app as a POST endpoint to the FastAPI router. Once all the webhooks are registered, the `run` method has to be
    called to start the app.

    It is recommended to accept [`WebhookPayload`] as the first argument of the webhook function. It is a Pydantic
    model that contains all the information about the webhook event. The data will be parsed automatically for you.

    Check out the [webhooks guide](../guides/webhooks_server) for a step-by-step tutorial on how to setup your
    WebhooksServer and deploy it on a Space.

    <Tip warning={true}>

    `WebhooksServer` is experimental. Its API is subject to change in the future.

    </Tip>

    <Tip warning={true}>

    You must have `gradio` installed to use `WebhooksServer` (`pip install --upgrade gradio`).

    </Tip>

    Args:
        ui (`gradio.Blocks`, optional):
            A Gradio UI instance to be used as the Space landing page. If `None`, a UI displaying instructions
            about the configured webhooks is created.
        webhook_secret (`str`, optional):
            A secret key to verify incoming webhook requests. You can set this value to any secret you want as long as
            you also configure it in your [webhooks settings panel](https://huggingface.co/settings/webhooks). You
            can also set this value as the `WEBHOOK_SECRET` environment variable. If no secret is provided, the
            webhook endpoints are opened without any security.

    Example:

        ```python
        import gradio as gr
        from huggingface_hub import WebhooksServer, WebhookPayload

        with gr.Blocks() as ui:
            ...

        app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

        @app.add_webhook("/say_hello")
        async def hello(payload: WebhookPayload):
            return {"message": "hello"}

        app.run()
        ```
    """

    def __new__(cls, *args, **kwargs) -> 'WebhooksServer':
        if not is_gradio_available():
            raise ImportError('You must have `gradio` installed to use `WebhooksServer`. Please run `pip install --upgrade gradio` first.')
        return super().__new__(cls)

    def __init__(self, ui: Optional['gr.Blocks']=None, webhook_secret: Optional[str]=None) -> None:
        self._ui = ui
        self.webhook_secret = webhook_secret or os.getenv('WEBHOOK_SECRET')
        self.registered_webhooks: Dict[str, Callable] = {}
        _warn_on_empty_secret(self.webhook_secret)

    def add_webhook(self, path: Optional[str]=None) -> Callable:
        """
        Decorator to add a webhook to the [`WebhooksServer`] server.

        Args:
            path (`str`, optional):
                The URL path to register the webhook function. If not provided, the function name will be used as the
                path. In any case, all webhooks are registered under `/webhooks`.

        Raises:
            ValueError: If the provided path is already registered as a webhook.

        Example:
            ```python
            from huggingface_hub import WebhooksServer, WebhookPayload

            app = WebhooksServer()

            @app.add_webhook
            async def trigger_training(payload: WebhookPayload):
                if payload.repo.type == "dataset" and payload.event.action == "update":
                    # Trigger a training job if a dataset is updated
                    ...

            app.run()
        ```
        """
        if callable(path):
            return self.add_webhook()(path)

        @wraps(FastAPI.post)
        def _inner_post(*args, **kwargs):
            func = args[0]
            abs_path = f'/webhooks/{(path or func.__name__).strip('/')}'
            if abs_path in self.registered_webhooks:
                raise ValueError(f'Webhook {abs_path} already exists.')
            self.registered_webhooks[abs_path] = func
        return _inner_post

    def launch(self, prevent_thread_lock: bool=False, **launch_kwargs: Any) -> None:
        """Launch the Gradio app and register webhooks to the underlying FastAPI server.

        Input parameters are forwarded to Gradio when launching the app.
        """
        ui = self._ui or self._get_default_ui()
        launch_kwargs.setdefault('share', _is_local)
        self.fastapi_app, _, _ = ui.launch(prevent_thread_lock=True, **launch_kwargs)
        for path, func in self.registered_webhooks.items():
            if self.webhook_secret is not None:
                func = _wrap_webhook_to_check_secret(func, webhook_secret=self.webhook_secret)
            self.fastapi_app.post(path)(func)
        url = (ui.share_url or ui.local_url).strip('/')
        message = '\nWebhooks are correctly setup and ready to use:'
        message += '\n' + '\n'.join((f'  - POST {url}{webhook}' for webhook in self.registered_webhooks))
        message += '\nGo to https://huggingface.co/settings/webhooks to setup your webhooks.'
        print(message)
        if not prevent_thread_lock:
            ui.block_thread()

    @_deprecate_method(version='0.23', message='Use `WebhooksServer.launch` instead.')
    def run(self) -> None:
        return self.launch()

    def _get_default_ui(self) -> 'gr.Blocks':
        """Default UI if not provided (lists webhooks and provides basic instructions)."""
        import gradio as gr
        with gr.Blocks() as ui:
            gr.Markdown('# This is an app to process 🤗 Webhooks')
            gr.Markdown('Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on specific repos or to all repos belonging to particular set of users/organizations (not just your repos, but any repo). Check out this [guide](https://huggingface.co/docs/hub/webhooks) to get to know more about webhooks on the Huggingface Hub.')
            gr.Markdown(f'{len(self.registered_webhooks)} webhook(s) are registered:' + '\n\n' + '\n '.join((f'- [{webhook_path}]({_get_webhook_doc_url(webhook.__name__, webhook_path)})' for webhook_path, webhook in self.registered_webhooks.items())))
            gr.Markdown('Go to https://huggingface.co/settings/webhooks to setup your webhooks.' + '\nYou app is running locally. Please look at the logs to check the full URL you need to set.' if _is_local else "\nThis app is running on a Space. You can find the corresponding URL in the options menu (top-right) > 'Embed the Space'. The URL looks like 'https://{username}-{repo_name}.hf.space'.")
        return ui