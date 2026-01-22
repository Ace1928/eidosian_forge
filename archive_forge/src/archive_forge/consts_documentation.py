from sentry_sdk._types import TYPE_CHECKING

    The "namespace" within which `code.function` is defined. Usually the qualified class or module name, such that `code.namespace` + some separator + `code.function` form a unique identifier for the code unit.
    Example: "http.handler"
    