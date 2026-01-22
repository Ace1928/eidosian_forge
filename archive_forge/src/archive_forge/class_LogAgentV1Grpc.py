import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (
class LogAgentV1Grpc(dashboard_utils.DashboardAgentModule):

    def __init__(self, dashboard_agent):
        super().__init__(dashboard_agent)

    async def run(self, server):
        if server:
            reporter_pb2_grpc.add_LogServiceServicer_to_server(self, server)

    @property
    def node_id(self) -> Optional[str]:
        return self._dashboard_agent.get_node_id()

    @staticmethod
    def is_minimal_module():
        return False

    async def ListLogs(self, request, context):
        """
        Lists all files in the active Ray logs directory.

        Part of `LogService` gRPC.

        NOTE: These RPCs are used by state_head.py, not log_head.py
        """
        path = Path(self._dashboard_agent.log_dir)
        if not path.exists():
            raise FileNotFoundError(f'Could not find log dir at path: {self._dashboard_agent.log_dir}It is unexpected. Please report an issue to Ray Github.')
        log_files = []
        for p in path.glob(request.glob_filter):
            log_files.append(str(p.relative_to(path)) + ('/' if p.is_dir() else ''))
        return reporter_pb2.ListLogsReply(log_files=log_files)

    @classmethod
    async def _find_task_log_offsets(cls, task_id: str, attempt_number: int, lines: int, f: io.BufferedIOBase) -> Tuple[int, int]:
        """Find the start and end offsets in the log file for a task attempt
        Current task log is in the format of below:

            :job_id:xxx
            :task_name:xxx
            :task_attempt_start:<task_id>-<attempt_number>
            ...
            actual user logs
            ...
            :task_attempt_end:<task_id>-<attempt_number>
            ... (other tasks)


        For async actor tasks, task logs from multiple tasks might however
        be interleaved.
        """
        task_attempt_start_magic_line = f'{LOG_PREFIX_TASK_ATTEMPT_START}{task_id}-{attempt_number}\n'
        task_attempt_magic_line_offset = await asyncio.get_running_loop().run_in_executor(_task_log_search_worker_pool, find_offset_of_content_in_file, f, task_attempt_start_magic_line.encode())
        if task_attempt_magic_line_offset == -1:
            raise FileNotFoundError(f'Log for task attempt({task_id},{attempt_number}) not found')
        start_offset = task_attempt_magic_line_offset + len(task_attempt_start_magic_line)
        task_attempt_end_magic_line = f'{LOG_PREFIX_TASK_ATTEMPT_END}{task_id}-{attempt_number}\n'
        end_offset = await asyncio.get_running_loop().run_in_executor(_task_log_search_worker_pool, find_offset_of_content_in_file, f, task_attempt_end_magic_line.encode(), start_offset)
        if end_offset == -1:
            end_offset = find_end_offset_file(f)
        if lines != -1:
            start_offset = max(find_start_offset_last_n_lines_from_offset(f, end_offset, lines), start_offset)
        return (start_offset, end_offset)

    @classmethod
    def _resolve_filename(cls, root_log_dir: Path, filename: str) -> Path:
        """
        Resolves the file path relative to the root log directory.

        Args:
            root_log_dir: Root log directory.
            filename: File path relative to the root log directory.

        Raises:
            FileNotFoundError: If the file path is invalid.

        Returns:
            The absolute file path resolved from the root log directory.
        """
        if not Path(filename).is_absolute():
            filepath = root_log_dir / filename
        else:
            filepath = Path(filename)
        filepath = Path(os.path.abspath(filepath))
        if not filepath.is_file():
            raise FileNotFoundError(f'A file is not found at: {filepath}')
        try:
            filepath.relative_to(root_log_dir)
        except ValueError as e:
            raise FileNotFoundError(f'{filepath} not in {root_log_dir}: {e}')
        return filepath.resolve()

    async def StreamLog(self, request, context):
        """
        Streams the log in real time starting from `request.lines` number of lines from
        the end of the file if `request.keep_alive == True`. Else, it terminates the
        stream once there are no more bytes to read from the log file.

        Part of `LogService` gRPC.

        NOTE: These RPCs are used by state_head.py, not log_head.py
        """
        lines = request.lines if request.lines else 1000
        try:
            filepath = self._resolve_filename(Path(self._dashboard_agent.log_dir), request.log_file_name)
        except FileNotFoundError as e:
            await context.send_initial_metadata([[log_consts.LOG_GRPC_ERROR, str(e)]])
        else:
            with open(filepath, 'rb') as f:
                await context.send_initial_metadata([])
                start_offset = request.start_offset if request.HasField('start_offset') else 0
                end_offset = request.end_offset if request.HasField('end_offset') else find_end_offset_file(f)
                if lines != -1:
                    start_offset = max(find_start_offset_last_n_lines_from_offset(f, offset=end_offset, n=lines), start_offset)
                keep_alive_interval_sec = -1
                if request.keep_alive:
                    keep_alive_interval_sec = request.interval if request.interval else DEFAULT_KEEP_ALIVE_INTERVAL_SEC
                    end_offset = -1
                logger.info(f'Tailing logs from {start_offset} to {end_offset} for lines={lines}, with keep_alive={keep_alive_interval_sec}')
                async for chunk_res in _stream_log_in_chunk(context=context, file=f, start_offset=start_offset, end_offset=end_offset, keep_alive_interval_sec=keep_alive_interval_sec):
                    yield chunk_res