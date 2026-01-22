from sentry_sdk.consts import TYPE_CHECKING
from sentry_sdk.db.explain_plan import cache_statement, should_run_explain_plan

    Run EXPLAIN or EXPLAIN ANALYZE on the given statement and attach the explain plan to the span data.

    Usage:
    ```
    sentry_sdk.init(
        dsn="...",
        _experiments={
            "attach_explain_plans": {
                "explain_cache_size": 1000,  # Run explain plan for the 1000 most run queries
                "explain_cache_timeout_seconds": 60 * 60 * 24,  # Run the explain plan for each statement only every 24 hours
                "use_explain_analyze": True,  # Run "explain analyze" instead of only "explain"
            }
        }
    ```
    