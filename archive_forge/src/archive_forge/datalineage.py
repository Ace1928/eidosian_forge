from datetime import datetime, timezone, timedelta
import uuid
from random import random
from openlineage.client import OpenLineageClient, OpenLineageClientOptions
from openlineage.client.run import RunEvent, RunState, Run, Job
from openlineage.client.facet import (
    SqlJobFacet,
    SchemaDatasetFacet,
    SchemaField,
    SourceCodeLocationJobFacet,
    NominalTimeRunFacet,
)

# Constants
PRODUCER_URL = "https://github.com/openlineage-user"
NAMESPACE = "python_client"
DAG_NAME = "user_trends"
API_URL = "http://mymarquez.host:5000"
API_KEY = "1234567890ckcu028rzu5l"

# Initialize OpenLineage client with API key
client = OpenLineageClient(
    url=API_URL, options=OpenLineageClientOptions(api_key=API_KEY)
)


def create_job(namespace: str, job_name: str, sql: str, location: str) -> Job:
    facets = {"sql": SqlJobFacet(sql)}
    if location:
        facets["sourceCodeLocation"] = SourceCodeLocationJobFacet("git", location)
    return Job(namespace=namespace, name=job_name, facets=facets)


def create_run(run_id: str, hour: int) -> Run:
    nominal_time = f"2022-04-14T{hour:02}:12:00Z"
    return Run(
        runId=run_id,
        facets={"nominalTime": NominalTimeRunFacet(nominalStartTime=nominal_time)},
    )


def create_dataset(
    namespace: str, name: str, schema: SchemaDatasetFacet = None
) -> dict:
    return {
        "namespace": namespace,
        "name": name,
        "facets": {"schema": schema} if schema else {},
    }


def create_run_event(
    job: Job,
    run: Run,
    event_type: RunState,
    event_time: str,
    inputs: list,
    outputs: list,
) -> RunEvent:
    return RunEvent(
        eventType=event_type,
        eventTime=event_time,
        run=run,
        job=job,
        producer=PRODUCER_URL,
        inputs=inputs,
        outputs=outputs,
    )


def schedule_run_events(
    job_name: str,
    sql: str,
    inputs: list,
    outputs: list,
    hour: int,
    minute: int,
    location: str,
    duration: int,
):
    run_id = str(uuid.uuid4())
    job = create_job(NAMESPACE, job_name, sql, location)
    run = create_run(run_id, hour)
    start_time = datetime.now(timezone.utc) + timedelta(hours=hour, minutes=minute)
    end_time = start_time + timedelta(minutes=duration)
    start_event = create_run_event(
        job, run, RunState.START, start_time.isoformat(), inputs, outputs
    )
    complete_event = create_run_event(
        job, run, RunState.COMPLETE, end_time.isoformat(), inputs, outputs
    )
    return start_event, complete_event


events = []
for i in range(5):
    schema_fields = [
        SchemaField(name="id", type="BIGINT", description="the user id"),
        SchemaField(
            name="email_domain", type="VARCHAR", description="the user email domain"
        ),
        SchemaField(name="status", type="BIGINT", description="the user status"),
        SchemaField(
            name="created_at", type="DATETIME", description="creation time of the user"
        ),
        SchemaField(
            name="updated_at",
            type="DATETIME",
            description="last update time of the user",
        ),
        SchemaField(
            name="fetch_time_utc", type="DATETIME", description="data fetch time"
        ),
        SchemaField(
            name="load_filename", type="VARCHAR", description="source file name"
        ),
        SchemaField(name="load_filerow", type="INT", description="source file row"),
        SchemaField(
            name="load_timestamp", type="DATETIME", description="data ingestion time"
        ),
    ]
    user_history_schema = SchemaDatasetFacet(fields=schema_fields)
    user_history = create_dataset(
        "snowflake://", "temp_demo.user_history", user_history_schema
    )
    user_counts = create_dataset(NAMESPACE, "tmp_demo.user_counts")

    sql_command = """
    CREATE OR REPLACE TABLE TMP_DEMO.USER_COUNTS AS (
        SELECT DATE_TRUNC('DAY', created_at) AS date, COUNT(id) AS user_count
        FROM TMP_DEMO.USER_HISTORY
        GROUP BY date
    )
    """
    location = "https://github.com/some/airflow/dags/example/user_trends.py"
    start_event, complete_event = schedule_run_events(
        f"{DAG_NAME}.create_user_counts",
        sql_command,
        [user_history],
        [user_counts],
        i,
        11,
        location,
        2,
    )
    events.extend([start_event, complete_event])

for event in events:
    client.emit(event)
