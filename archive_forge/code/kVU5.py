import random
import logging
from pygame.math import Vector2
from Constants import BANNER_HEIGHT, NO_OF_CELLS, USER_SEED
from typing import Tuple, NoReturn
import asyncio

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Re-seed the random number generator to ensure reproducibility
random.seed(USER_SEED)
logging.debug(f"Random seed set to {USER_SEED} ensuring reproducibility across runs.")


class Fruit:
    def __init__(self) -> None:
        """
        Initializes the Fruit object by setting its initial position to the origin (0, 0)
        and then resetting the seed to generate the fruit's position.
        """
        self.position: Vector2 = Vector2(0, 0)
        logging.debug("Fruit object initialized with position set to Vector2(0, 0).")
        self.lock = asyncio.Lock()
        asyncio.create_task(self.reset_seed())

    async def generate_fruit(self) -> None:
        """
        Generates a new position for the fruit within the defined game boundaries,
        avoiding the outermost cells and the banner area, using asynchronous operations.
        """
        border: int = NO_OF_CELLS - 1
        logging.debug(f"Calculated border as {border} for fruit generation.")

        async def generate_position():
            x: int = random.randrange(1, border)
            y: int = random.randrange(BANNER_HEIGHT, border)
            return Vector2(x, y)

        async with self.lock:
            self.position = await generate_position()
            logging.debug(f"Fruit position updated to {self.position}.")

    async def reset_seed(self) -> None:
        """
        Resets the random seed to ensure consistent random behavior and then generates
        the fruit's position asynchronously.
        """
        async with self.lock:
            random.seed(USER_SEED)
            logging.debug(f"Random seed reset to {USER_SEED}.")
            await self.generate_fruit()
            logging.debug("Fruit position regenerated after seed reset.")
