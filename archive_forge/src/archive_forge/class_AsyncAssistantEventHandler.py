from __future__ import annotations
import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Callable, Iterable, Iterator, cast
from typing_extensions import Awaitable, AsyncIterable, AsyncIterator, assert_never
import httpx
from ..._utils import is_dict, is_list, consume_sync_iterator, consume_async_iterator
from ..._models import construct_type
from ..._streaming import Stream, AsyncStream
from ...types.beta import AssistantStreamEvent
from ...types.beta.threads import (
from ...types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta
class AsyncAssistantEventHandler:
    text_deltas: AsyncIterable[str]
    'Iterator over just the text deltas in the stream.\n\n    This corresponds to the `thread.message.delta` event\n    in the API.\n\n    ```py\n    async for text in stream.text_deltas:\n        print(text, end="", flush=True)\n    print()\n    ```\n    '

    def __init__(self) -> None:
        self._current_event: AssistantStreamEvent | None = None
        self._current_message_content_index: int | None = None
        self._current_message_content: MessageContent | None = None
        self._current_tool_call_index: int | None = None
        self._current_tool_call: ToolCall | None = None
        self.__current_run_step_id: str | None = None
        self.__current_run: Run | None = None
        self.__run_step_snapshots: dict[str, RunStep] = {}
        self.__message_snapshots: dict[str, Message] = {}
        self.__current_message_snapshot: Message | None = None
        self.text_deltas = self.__text_deltas__()
        self._iterator = self.__stream__()
        self.__stream: AsyncStream[AssistantStreamEvent] | None = None

    def _init(self, stream: AsyncStream[AssistantStreamEvent]) -> None:
        if self.__stream:
            raise RuntimeError('A single event handler cannot be shared between multiple streams; You will need to construct a new event handler instance')
        self.__stream = stream

    async def __anext__(self) -> AssistantStreamEvent:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[AssistantStreamEvent]:
        async for item in self._iterator:
            yield item

    async def close(self) -> None:
        """
        Close the response and release the connection.

        Automatically called when the context manager exits.
        """
        if self.__stream:
            await self.__stream.close()

    @property
    def current_event(self) -> AssistantStreamEvent | None:
        return self._current_event

    @property
    def current_run(self) -> Run | None:
        return self.__current_run

    @property
    def current_run_step_snapshot(self) -> RunStep | None:
        if not self.__current_run_step_id:
            return None
        return self.__run_step_snapshots[self.__current_run_step_id]

    @property
    def current_message_snapshot(self) -> Message | None:
        return self.__current_message_snapshot

    async def until_done(self) -> None:
        """Waits until the stream has been consumed"""
        await consume_async_iterator(self)

    async def get_final_run(self) -> Run:
        """Wait for the stream to finish and returns the completed Run object"""
        await self.until_done()
        if not self.__current_run:
            raise RuntimeError('No final run object found')
        return self.__current_run

    async def get_final_run_steps(self) -> list[RunStep]:
        """Wait for the stream to finish and returns the steps taken in this run"""
        await self.until_done()
        if not self.__run_step_snapshots:
            raise RuntimeError('No run steps found')
        return [step for step in self.__run_step_snapshots.values()]

    async def get_final_messages(self) -> list[Message]:
        """Wait for the stream to finish and returns the messages emitted in this run"""
        await self.until_done()
        if not self.__message_snapshots:
            raise RuntimeError('No messages found')
        return [message for message in self.__message_snapshots.values()]

    async def __text_deltas__(self) -> AsyncIterator[str]:
        async for event in self:
            if event.event != 'thread.message.delta':
                continue
            for content_delta in event.data.delta.content or []:
                if content_delta.type == 'text' and content_delta.text and content_delta.text.value:
                    yield content_delta.text.value

    async def on_end(self) -> None:
        """Fires when the stream has finished.

        This happens if the stream is read to completion
        or if an exception occurs during iteration.
        """

    async def on_event(self, event: AssistantStreamEvent) -> None:
        """Callback that is fired for every Server-Sent-Event"""

    async def on_run_step_created(self, run_step: RunStep) -> None:
        """Callback that is fired when a run step is created"""

    async def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep) -> None:
        """Callback that is fired whenever a run step delta is returned from the API

        The first argument is just the delta as sent by the API and the second argument
        is the accumulated snapshot of the run step. For example, a tool calls event may
        look like this:

        # delta
        tool_calls=[
            RunStepDeltaToolCallsCodeInterpreter(
                index=0,
                type='code_interpreter',
                id=None,
                code_interpreter=CodeInterpreter(input=' sympy', outputs=None)
            )
        ]
        # snapshot
        tool_calls=[
            CodeToolCall(
                id='call_wKayJlcYV12NiadiZuJXxcfx',
                code_interpreter=CodeInterpreter(input='from sympy', outputs=[]),
                type='code_interpreter',
                index=0
            )
        ],
        """

    async def on_run_step_done(self, run_step: RunStep) -> None:
        """Callback that is fired when a run step is completed"""

    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is created"""

    async def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall) -> None:
        """Callback that is fired when a tool call delta is encountered"""

    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call delta is encountered"""

    async def on_exception(self, exception: Exception) -> None:
        """Fired whenever an exception happens during streaming"""

    async def on_timeout(self) -> None:
        """Fires if the request times out"""

    async def on_message_created(self, message: Message) -> None:
        """Callback that is fired when a message is created"""

    async def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
        """Callback that is fired whenever a message delta is returned from the API

        The first argument is just the delta as sent by the API and the second argument
        is the accumulated snapshot of the message. For example, a text content event may
        look like this:

        # delta
        MessageDeltaText(
            index=0,
            type='text',
            text=Text(
                value=' Jane'
            ),
        )
        # snapshot
        MessageContentText(
            index=0,
            type='text',
            text=Text(
                value='Certainly, Jane'
            ),
        )
        """

    async def on_message_done(self, message: Message) -> None:
        """Callback that is fired when a message is completed"""

    async def on_text_created(self, text: Text) -> None:
        """Callback that is fired when a text content block is created"""

    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        """Callback that is fired whenever a text content delta is returned
        by the API.

        The first argument is just the delta as sent by the API and the second argument
        is the accumulated snapshot of the text. For example:

        on_text_delta(TextDelta(value="The"), Text(value="The")),
        on_text_delta(TextDelta(value=" solution"), Text(value="The solution")),
        on_text_delta(TextDelta(value=" to"), Text(value="The solution to")),
        on_text_delta(TextDelta(value=" the"), Text(value="The solution to the")),
        on_text_delta(TextDelta(value=" equation"), Text(value="The solution to the equivalent")),
        """

    async def on_text_done(self, text: Text) -> None:
        """Callback that is fired when a text content block is finished"""

    async def on_image_file_done(self, image_file: ImageFile) -> None:
        """Callback that is fired when an image file block is finished"""

    async def _emit_sse_event(self, event: AssistantStreamEvent) -> None:
        self._current_event = event
        await self.on_event(event)
        self.__current_message_snapshot, new_content = accumulate_event(event=event, current_message_snapshot=self.__current_message_snapshot)
        if self.__current_message_snapshot is not None:
            self.__message_snapshots[self.__current_message_snapshot.id] = self.__current_message_snapshot
        accumulate_run_step(event=event, run_step_snapshots=self.__run_step_snapshots)
        for content_delta in new_content:
            assert self.__current_message_snapshot is not None
            block = self.__current_message_snapshot.content[content_delta.index]
            if block.type == 'text':
                await self.on_text_created(block.text)
        if event.event == 'thread.run.completed' or event.event == 'thread.run.cancelled' or event.event == 'thread.run.expired' or (event.event == 'thread.run.failed') or (event.event == 'thread.run.requires_action'):
            self.__current_run = event.data
            if self._current_tool_call:
                await self.on_tool_call_done(self._current_tool_call)
        elif event.event == 'thread.run.created' or event.event == 'thread.run.in_progress' or event.event == 'thread.run.cancelling' or (event.event == 'thread.run.queued'):
            self.__current_run = event.data
        elif event.event == 'thread.message.created':
            await self.on_message_created(event.data)
        elif event.event == 'thread.message.delta':
            snapshot = self.__current_message_snapshot
            assert snapshot is not None
            message_delta = event.data.delta
            if message_delta.content is not None:
                for content_delta in message_delta.content:
                    if content_delta.type == 'text' and content_delta.text:
                        snapshot_content = snapshot.content[content_delta.index]
                        assert snapshot_content.type == 'text'
                        await self.on_text_delta(content_delta.text, snapshot_content.text)
                    if content_delta.index != self._current_message_content_index:
                        if self._current_message_content is not None:
                            if self._current_message_content.type == 'text':
                                await self.on_text_done(self._current_message_content.text)
                            elif self._current_message_content.type == 'image_file':
                                await self.on_image_file_done(self._current_message_content.image_file)
                        self._current_message_content_index = content_delta.index
                        self._current_message_content = snapshot.content[content_delta.index]
                    self._current_message_content = snapshot.content[content_delta.index]
            await self.on_message_delta(event.data.delta, snapshot)
        elif event.event == 'thread.message.completed' or event.event == 'thread.message.incomplete':
            self.__current_message_snapshot = event.data
            self.__message_snapshots[event.data.id] = event.data
            if self._current_message_content_index is not None:
                content = event.data.content[self._current_message_content_index]
                if content.type == 'text':
                    await self.on_text_done(content.text)
                elif content.type == 'image_file':
                    await self.on_image_file_done(content.image_file)
            await self.on_message_done(event.data)
        elif event.event == 'thread.run.step.created':
            self.__current_run_step_id = event.data.id
            await self.on_run_step_created(event.data)
        elif event.event == 'thread.run.step.in_progress':
            self.__current_run_step_id = event.data.id
        elif event.event == 'thread.run.step.delta':
            step_snapshot = self.__run_step_snapshots[event.data.id]
            run_step_delta = event.data.delta
            if run_step_delta.step_details and run_step_delta.step_details.type == 'tool_calls' and (run_step_delta.step_details.tool_calls is not None):
                assert step_snapshot.step_details.type == 'tool_calls'
                for tool_call_delta in run_step_delta.step_details.tool_calls:
                    if tool_call_delta.index == self._current_tool_call_index:
                        await self.on_tool_call_delta(tool_call_delta, step_snapshot.step_details.tool_calls[tool_call_delta.index])
                    if tool_call_delta.index != self._current_tool_call_index:
                        if self._current_tool_call is not None:
                            await self.on_tool_call_done(self._current_tool_call)
                        self._current_tool_call_index = tool_call_delta.index
                        self._current_tool_call = step_snapshot.step_details.tool_calls[tool_call_delta.index]
                        await self.on_tool_call_created(self._current_tool_call)
                    self._current_tool_call = step_snapshot.step_details.tool_calls[tool_call_delta.index]
            await self.on_run_step_delta(event.data.delta, step_snapshot)
        elif event.event == 'thread.run.step.completed' or event.event == 'thread.run.step.cancelled' or event.event == 'thread.run.step.expired' or (event.event == 'thread.run.step.failed'):
            if self._current_tool_call:
                await self.on_tool_call_done(self._current_tool_call)
            await self.on_run_step_done(event.data)
            self.__current_run_step_id = None
        elif event.event == 'thread.created' or event.event == 'thread.message.in_progress' or event.event == 'error':
            ...
        elif TYPE_CHECKING:
            assert_never(event)
        self._current_event = None

    async def __stream__(self) -> AsyncIterator[AssistantStreamEvent]:
        stream = self.__stream
        if not stream:
            raise RuntimeError('Stream has not been started yet')
        try:
            async for event in stream:
                await self._emit_sse_event(event)
                yield event
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            await self.on_timeout()
            await self.on_exception(exc)
            raise
        except Exception as exc:
            await self.on_exception(exc)
            raise
        finally:
            await self.on_end()