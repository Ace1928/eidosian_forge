import time
import warnings
from typing import Any
from traitlets import Instance
from ..restarter import KernelRestarter
class AsyncIOLoopKernelRestarter(IOLoopKernelRestarter):
    """An async io loop kernel restarter."""

    async def poll(self) -> None:
        """Poll the kernel."""
        if self.debug:
            self.log.debug('Polling kernel...')
        is_alive = await self.kernel_manager.is_alive()
        now = time.time()
        if not is_alive:
            self._last_dead = now
            if self._restarting:
                self._restart_count += 1
            else:
                self._restart_count = 1
            if self._restart_count > self.restart_limit:
                self.log.warning('AsyncIOLoopKernelRestarter: restart failed')
                self._fire_callbacks('dead')
                self._restarting = False
                self._restart_count = 0
                self.stop()
            else:
                newports = self.random_ports_until_alive and self._initial_startup
                self.log.info('AsyncIOLoopKernelRestarter: restarting kernel (%i/%i), %s random ports', self._restart_count, self.restart_limit, 'new' if newports else 'keep')
                self._fire_callbacks('restart')
                await self.kernel_manager.restart_kernel(now=True, newports=newports)
                self._restarting = True
        else:
            stable_start_time = self.stable_start_time
            if self.kernel_manager.provisioner:
                stable_start_time = self.kernel_manager.provisioner.get_stable_start_time(recommended=stable_start_time)
            if self._initial_startup and now - self._last_dead >= stable_start_time:
                self._initial_startup = False
            if self._restarting and now - self._last_dead >= stable_start_time:
                self.log.debug('AsyncIOLoopKernelRestarter: restart apparently succeeded')
                self._restarting = False