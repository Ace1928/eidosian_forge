import dataclasses
import inspect
import logging
from collections import defaultdict
from functools import wraps
from typing import List, Optional, Tuple
import aiohttp
import grpc
from grpc.aio._call import UnaryStreamCall
import ray
import ray.dashboard.modules.log.log_consts as log_consts
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import hex_to_binary
from ray._raylet import ActorID, JobID, TaskID
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated.gcs_pb2 import ActorTableData
from ray.core.generated.gcs_service_pb2 import (
from ray.core.generated.node_manager_pb2 import (
from ray.core.generated.node_manager_pb2_grpc import NodeManagerServiceStub
from ray.core.generated.reporter_pb2 import (
from ray.core.generated.reporter_pb2_grpc import LogServiceStub
from ray.core.generated.runtime_env_agent_pb2 import (
from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.job.common import JobInfoStorageClient
from ray.dashboard.modules.job.pydantic_models import JobDetails, JobType
from ray.dashboard.modules.job.utils import get_driver_jobs
from ray.dashboard.utils import Dict as Dictionary
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
class StateDataSourceClient:
    """The client to query states from various data sources such as Raylet, GCS, Agents.

    Note that it doesn't directly query core workers. They are proxied through raylets.

    The module is not in charge of service discovery. The caller is responsible for
    finding services and register stubs through `register*` APIs.

    Non `register*` APIs
    - Return the protobuf directly if it succeeds to query the source.
    - Raises an exception if there's any network issue.
    - throw a ValueError if it cannot find the source.
    """

    def __init__(self, gcs_channel: grpc.aio.Channel, gcs_aio_client: GcsAioClient):
        self.register_gcs_client(gcs_channel)
        self._raylet_stubs = {}
        self._runtime_env_agent_addresses = {}
        self._log_agent_stub = {}
        self._job_client = JobInfoStorageClient(gcs_aio_client)
        self._id_id_map = IdToIpMap()
        self._gcs_aio_client = gcs_aio_client
        self._client_session = aiohttp.ClientSession()

    def register_gcs_client(self, gcs_channel: grpc.aio.Channel):
        self._gcs_actor_info_stub = gcs_service_pb2_grpc.ActorInfoGcsServiceStub(gcs_channel)
        self._gcs_pg_info_stub = gcs_service_pb2_grpc.PlacementGroupInfoGcsServiceStub(gcs_channel)
        self._gcs_node_info_stub = gcs_service_pb2_grpc.NodeInfoGcsServiceStub(gcs_channel)
        self._gcs_worker_info_stub = gcs_service_pb2_grpc.WorkerInfoGcsServiceStub(gcs_channel)
        self._gcs_task_info_stub = gcs_service_pb2_grpc.TaskInfoGcsServiceStub(gcs_channel)

    def register_raylet_client(self, node_id: str, address: str, port: int, runtime_env_agent_port: int):
        full_addr = f'{address}:{port}'
        options = _STATE_MANAGER_GRPC_OPTIONS
        channel = ray._private.utils.init_grpc_channel(full_addr, options, asynchronous=True)
        self._raylet_stubs[node_id] = NodeManagerServiceStub(channel)
        self._runtime_env_agent_addresses[node_id] = f'http://{address}:{runtime_env_agent_port}'
        self._id_id_map.put(node_id, address)

    def unregister_raylet_client(self, node_id: str):
        self._raylet_stubs.pop(node_id)
        self._runtime_env_agent_addresses.pop(node_id)
        self._id_id_map.pop(node_id)

    def register_agent_client(self, node_id, address: str, port: int):
        options = _STATE_MANAGER_GRPC_OPTIONS
        channel = ray._private.utils.init_grpc_channel(f'{address}:{port}', options=options, asynchronous=True)
        self._log_agent_stub[node_id] = LogServiceStub(channel)
        self._id_id_map.put(node_id, address)

    def unregister_agent_client(self, node_id: str):
        self._log_agent_stub.pop(node_id)
        self._id_id_map.pop(node_id)

    def get_all_registered_raylet_ids(self) -> List[str]:
        return self._raylet_stubs.keys()

    def get_all_registered_runtime_env_agent_ids(self) -> List[str]:
        return self._runtime_env_agent_addresses.keys()

    def get_all_registered_log_agent_ids(self) -> List[str]:
        return self._log_agent_stub.keys()

    def ip_to_node_id(self, ip: Optional[str]) -> Optional[str]:
        """Return the node id that corresponds to the given ip.

        Args:
            ip: The ip address.

        Returns:
            None if the corresponding id doesn't exist.
            Node id otherwise. If None node_ip is given,
            it will also return None.
        """
        if not ip:
            return None
        return self._id_id_map.get_node_id(ip)

    @handle_grpc_network_errors
    async def get_all_actor_info(self, timeout: int=None, limit: int=None, filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]]=None) -> Optional[GetAllActorInfoReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        if filters is None:
            filters = []
        req_filters = GetAllActorInfoRequest.Filters()
        for filter in filters:
            key, predicate, value = filter
            if predicate != '=':
                continue
            if key == 'actor_id':
                req_filters.actor_id = ActorID(hex_to_binary(value)).binary()
            elif key == 'state':
                value = value.upper()
                if value not in ActorTableData.ActorState.keys():
                    raise ValueError(f'Invalid actor state for filtering: {value}')
                req_filters.state = ActorTableData.ActorState.Value(value)
            elif key == 'job_id':
                req_filters.job_id = JobID(hex_to_binary(value)).binary()
        request = GetAllActorInfoRequest(limit=limit, filters=req_filters)
        reply = await self._gcs_actor_info_stub.GetAllActorInfo(request, timeout=timeout)
        return reply

    @handle_grpc_network_errors
    async def get_all_task_info(self, timeout: int=None, limit: int=None, filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]]=None, exclude_driver: bool=False) -> Optional[GetTaskEventsReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        if filters is None:
            filters = []
        req_filters = GetTaskEventsRequest.Filters()
        for filter in filters:
            key, predicate, value = filter
            if predicate != '=':
                continue
            if key == 'actor_id':
                req_filters.actor_id = ActorID(hex_to_binary(value)).binary()
            elif key == 'job_id':
                req_filters.job_id = JobID(hex_to_binary(value)).binary()
            elif key == 'task_id':
                req_filters.task_ids.append(TaskID(hex_to_binary(value)).binary())
            else:
                continue
        req_filters.exclude_driver = exclude_driver
        request = GetTaskEventsRequest(limit=limit, filters=req_filters)
        reply = await self._gcs_task_info_stub.GetTaskEvents(request, timeout=timeout)
        return reply

    @handle_grpc_network_errors
    async def get_all_placement_group_info(self, timeout: int=None, limit: int=None) -> Optional[GetAllPlacementGroupReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        request = GetAllPlacementGroupRequest(limit=limit)
        reply = await self._gcs_pg_info_stub.GetAllPlacementGroup(request, timeout=timeout)
        return reply

    @handle_grpc_network_errors
    async def get_all_node_info(self, timeout: int=None) -> Optional[GetAllNodeInfoReply]:
        request = GetAllNodeInfoRequest()
        reply = await self._gcs_node_info_stub.GetAllNodeInfo(request, timeout=timeout)
        return reply

    @handle_grpc_network_errors
    async def get_all_worker_info(self, timeout: int=None, limit: int=None) -> Optional[GetAllWorkerInfoReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        request = GetAllWorkerInfoRequest(limit=limit)
        reply = await self._gcs_worker_info_stub.GetAllWorkerInfo(request, timeout=timeout)
        return reply

    async def get_job_info(self, timeout: int=None) -> List[JobDetails]:
        driver_jobs, submission_job_drivers = await get_driver_jobs(self._gcs_aio_client, timeout=timeout)
        submission_jobs = await self._job_client.get_all_jobs(timeout=timeout)
        submission_jobs = [JobDetails(**dataclasses.asdict(job), submission_id=submission_id, job_id=submission_job_drivers.get(submission_id).id if submission_id in submission_job_drivers else None, driver_info=submission_job_drivers.get(submission_id), type=JobType.SUBMISSION) for submission_id, job in submission_jobs.items()]
        return list(driver_jobs.values()) + submission_jobs

    async def get_all_cluster_events(self) -> Dictionary:
        return DataSource.events

    @handle_grpc_network_errors
    async def get_task_info(self, node_id: str, timeout: int=None, limit: int=None) -> Optional[GetTasksInfoReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        stub = self._raylet_stubs.get(node_id)
        if not stub:
            raise ValueError(f"Raylet for a node id, {node_id} doesn't exist.")
        reply = await stub.GetTasksInfo(GetTasksInfoRequest(limit=limit), timeout=timeout)
        return reply

    @handle_grpc_network_errors
    async def get_object_info(self, node_id: str, timeout: int=None, limit: int=None) -> Optional[GetObjectsInfoReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        stub = self._raylet_stubs.get(node_id)
        if not stub:
            raise ValueError(f"Raylet for a node id, {node_id} doesn't exist.")
        reply = await stub.GetObjectsInfo(GetObjectsInfoRequest(limit=limit), timeout=timeout)
        return reply

    async def get_runtime_envs_info(self, node_id: str, timeout: int=None, limit: int=None) -> Optional[GetRuntimeEnvsInfoReply]:
        if not limit:
            limit = RAY_MAX_LIMIT_FROM_DATA_SOURCE
        address = self._runtime_env_agent_addresses.get(node_id)
        if not address:
            raise ValueError(f"Runtime Env Agent for a node id, {node_id} doesn't exist.")
        timeout = aiohttp.ClientTimeout(total=timeout)
        url = f'{address}/get_runtime_envs_info'
        request = GetRuntimeEnvsInfoRequest(limit=limit)
        data = request.SerializeToString()
        async with self._client_session.post(url, data=data, timeout=timeout) as resp:
            if resp.status >= 200 and resp.status < 300:
                response_data = await resp.read()
                reply = GetRuntimeEnvsInfoReply()
                reply.ParseFromString(response_data)
                return reply
            else:
                raise DataSourceUnavailable(f"Failed to query the runtime env agent for get_runtime_envs_info. Either there's a network issue, or the source is down. Response is {resp.status}, reason {resp.reason}")

    @handle_grpc_network_errors
    async def list_logs(self, node_id: str, glob_filter: str, timeout: int=None) -> ListLogsReply:
        stub = self._log_agent_stub.get(node_id)
        if not stub:
            raise ValueError(f"Agent for node id: {node_id} doesn't exist.")
        return await stub.ListLogs(ListLogsRequest(glob_filter=glob_filter), timeout=timeout)

    @handle_grpc_network_errors
    async def stream_log(self, node_id: str, log_file_name: str, keep_alive: bool, lines: int, interval: Optional[float], timeout: int, start_offset: Optional[int]=None, end_offset: Optional[int]=None) -> UnaryStreamCall:
        stub = self._log_agent_stub.get(node_id)
        if not stub:
            raise ValueError(f"Agent for node id: {node_id} doesn't exist.")
        stream = stub.StreamLog(StreamLogRequest(keep_alive=keep_alive, log_file_name=log_file_name, lines=lines, interval=interval, start_offset=start_offset, end_offset=end_offset), timeout=timeout)
        metadata = await stream.initial_metadata()
        if metadata.get(log_consts.LOG_GRPC_ERROR) is not None:
            raise ValueError(metadata.get(log_consts.LOG_GRPC_ERROR))
        return stream