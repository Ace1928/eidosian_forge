from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNetwork
import pickle
import logging
import asyncio
import aiofiles
from DecisionTree import DecisionTree


# Configure logging to the highest level of verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Snake:
    """
    A class representing the Snake entity in a Snake game, equipped with AI capabilities.

    Attributes:
        body (list[Vector2]): A list of Vector2 objects representing the segments of the snake's body.
        fruit (Fruit): An instance of the Fruit class representing the fruit object in the game.
        score (int): The current score of the snake in the game.
        fitness (float): A metric representing the snake's performance.
        life_time (int): The total duration the snake has been alive in the game.
        steps (int): The number of steps the snake has taken.
        hidden (int): The number of hidden layers in the neural network.
        network (NeuralNetwork): An instance of the NeuralNetwork class used for AI decision making.
    """

    def __init__(self, hidden: int = 20) -> None:
        """
        Initializes the Snake class with a specified number of hidden layers for the neural network.

        Args:
            hidden (int): The number of hidden layers in the neural network.
        """
        logging.debug("Initializing the Snake class with meticulous detail.")
        self._body = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        self._fruit = Fruit()
        self._score = 0
        self._fitness = 0.0
        self._life_time = 0
        self._steps = 0
        self._hidden = hidden
        self._network = NeuralNetwork(5, self._hidden, 3)
        logging.debug(
            "Snake class initialized successfully with parameters: body={}, fruit={}, score={}, fitness={}, life_time={}, steps={}, hidden={}, network={}".format(
                self._body,
                self._fruit,
                self._score,
                self._fitness,
                self._life_time,
                self._steps,
                self._hidden,
                self._network,
            )
        )

    async def save_model(self, network: NeuralNetwork, name: str) -> None:
        """
        Asynchronously saves the neural network model to a file.

        Args:
            network (NeuralNetwork): The neural network to be saved.
            name (str): The filename to save the model to.

        Raises:
            RuntimeError: If the model fails to save.
        """
        logging.debug(
            f"Attempting to save the model to {name} with detailed error handling."
        )
        try:
            async with aiofiles.open(name, "wb") as file:
                await file.write(pickle.dumps(network))
            logging.info(f"Model saved successfully in {name}.")
        except Exception as e:
            logging.error(f"Failed to save the model to {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save the model due to: {e}") from e

    async def load_model(self, name: str) -> None:
        """
        Asynchronously loads the neural network model from a file.

        Args:
            name (str): The filename to load the model from.

        Raises:
            RuntimeError: If the model fails to load.
        """
        logging.debug(
            f"Attempting to load the model from {name} with detailed error handling."
        )
        try:
            async with aiofiles.open(name, "rb") as file:
                self.network = pickle.loads(await file.read())
            logging.info(f"Model loaded successfully from {name}.")
        except Exception as e:
            logging.error(f"Failed to load the model from {name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load the model due to: {e}") from e

    def reset(self) -> None:
        """
        Resets the snake to its initial state.
        """
        logging.debug(
            "Resetting the Snake instance to its initial state with detailed logging."
        )
        self.body = [Vector2(5, 8), Vector2(4, 8), Vector2(3, 8)]
        self.fruit.reset_seed()
        self.score = 0
        self.fitness = 0
        self.steps = 0
        self.network = NeuralNetwork(5, self.hidden, 3)
        logging.info(
            "Snake instance has been reset successfully with parameters reset to their initial values."
        )

    @property
    def body(self) -> list[Vector2]:
        """
        list[Vector2]: The body of the snake.
        """
        return self._body

    @body.setter
    def body(self, value: list[Vector2]) -> None:
        """
        Sets the body of the snake.

        Args:
            value (list[Vector2]): The body of the snake.
        """
        self._body = value
        logging.debug(f"Setting the body of the snake to {value}.")

    @property
    def fruit(self) -> Fruit:
        """
        Fruit: The fruit object in the game.
        """
        return self._fruit

    @fruit.setter
    def fruit(self, value: Fruit) -> None:
        """
        Sets the fruit object in the game.

        Args:
            value (Fruit): The fruit object.
        """
        self._fruit = value
        logging.debug(f"Setting the fruit object to {value}.")

    @property
    def score(self) -> int:
        """
        int: The score of the snake.
        """
        return self._score

    @score.setter
    def score(self, value: int) -> None:
        """
        Sets the score of the snake.

        Args:
            value (int): The score of the snake.
        """
        self._score = value
        logging.debug(f"Setting the score of the snake to {value}.")

    @property
    def fitness(self) -> float:
        """
        float: The fitness value of the snake.
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        """
        Sets the fitness value of the snake.

        Args:
            value (float): The fitness value of the snake.
        """
        self._fitness = value
        logging.debug(f"Setting the fitness value of the snake to {value}.")

    @property
    def life_time(self) -> int:
        """
        int: The life time of the snake.
        """
        return self._life_time

    @life_time.setter
    def life_time(self, value: int) -> None:
        """
        Sets the life time of the snake.

        Args:
            value (int): The life time of the snake.

        Raises:
            ValueError: If the life time is negative.
        """
        if value < 0:
            raise ValueError("The life time of the snake cannot be negative.")
        self._life_time = value
        logging.debug(f"Setting the life time of the snake to {value}.")

    def get_x(self) -> float:
        """
        Retrieves the x-coordinate of the snake's head.
        Returns:
            float: The x-coordinate of the snake's head.
        """
        logging.debug("Retrieving the x-coordinate of the snake's head.")
        return self.body[0].x

    def get_y(self) -> float:
        """
        Retrieves the y-coordinate of the snake's head.
        Returns:
            float: The y-coordinate of the snake's head.
        """
        logging.debug("Retrieving the y-coordinate of the snake's head.")
        return self.body[0].y

    def get_fruit(self) -> Vector2:
        """
        Retrieves the position of the fruit.
        Returns:
            Vector2: The position of the fruit.
        """
        logging.debug("Retrieving the position of the fruit.")
        return self.fruit.position

    def ate_fruit(self) -> bool:
        """
        Checks if the snake has eaten the fruit.
        Returns:
            bool: True if the snake has eaten the fruit, False otherwise.
        """
        logging.debug("Checking if the snake has eaten the fruit.")
        if self.fruit.position == self.body[0]:
            self.score += 1
            self.life_time -= 40
            logging.info(
                "Fruit has been eaten. Score incremented and life_time decremented."
            )
            return True
        return False

    def create_fruit(self) -> None:
        """
        Creates a new fruit in the game.
        """
        logging.debug("Creating a new fruit with detailed logging.")
        self.fruit.generate_fruit()
        logging.info("New fruit created successfully.")

    def move_ai(self, x: float, y: float) -> None:
        """
        Moves the AI snake based on provided coordinates.
        Args:
            x (float): The x-coordinate to move to.
            y (float): The y-coordinate to move to.
        """
        logging.debug("Moving AI based on provided coordinates with detailed logging.")
        self.life_time += 1
        self.steps += 1
        for i in range(len(self.body) - 1, 0, -1):
            self.body[i].x = self.body[i - 1].x
            self.body[i].y = self.body[i - 1].y

        self.body[0].x = x
        self.body[0].y = y
        logging.info("AI moved successfully with updated coordinates.")

    def add_body_ai(self) -> None:
        """
        Adds a new body segment to the AI snake.
        """
        logging.debug(
            "Adding a new body segment to the AI snake with detailed logging."
        )
        last_index: int = len(self.body) - 1
        tail: Vector2 = self.body[-1]
        before_last: Vector2 = self.body[-2]

        if tail.x == before_last.x:
            if tail.y < before_last.y:
                self.body.append(Vector2(tail.x, tail.y - 1))
            else:
                self.body.append(Vector2(tail.x, tail.y + 1))
        elif tail.y == before_last.y:
            if tail.x < before_last.x:
                self.body.append(Vector2(tail.x - 1, tail.y))
            else:
                self.body.append(Vector2(tail.x + 1, tail.y))
        logging.info(
            "New body segment added successfully with detailed position logging."
        )

    def ate_body(self) -> bool:
        """
        Checks if the snake has collided with its own body.

        Returns:
            bool: True if the snake has collided with its body, False otherwise.
        """
        logging.debug(
            "Checking if the snake has collided with its body with detailed logging."
        )
        for body_part in self.body[1:]:
            if self.body[0] == body_part:
                logging.info("Collision detected: Snake has eaten its body.")
                return True
        return False

    def predict_move(self) -> int:
        """
        Predicts the next move for the AI snake based on its vision.

        Returns:
            int: The predicted move for the AI snake.
        """
        logging.debug("Predicting the next move for the AI snake based on its vision.")
        return self.decision_tree.predict(self.vision)

    def predict_vision(self) -> List[float]:
        """
        Predicts the vision for the AI snake.

        Returns:
            List[float]: The predicted vision for the AI snake.
        """
        logging.debug("Predicting the vision for the AI snake.")
        return self.decision_tree.predict(self.vision)

    @property
    def head(self) -> Vector2:
        """
        Vector2: The position of the snake's head.
        """
        return self.body[0]

    @property
    def tail(self) -> Vector2:
        """
        Vector2: The position of the snake's tail.
        """
        return self.body[-1]

    @property
    def direction(self) -> Vector2:
        """
        Vector2: The direction of the snake's movement.
        """
        return self.body[0] - self.body[1]

    @property
    def distance_to_fruit(self) -> Vector2:
        """
        Vector2: The distance vector from the snake's head to the fruit.
        """
        return self.fruit.position - self.body[0]

    @property
    def angle_to_fruit(self) -> float:
        """
        float: The angle in radians between the snake's head and the fruit.
        """
        return self.direction.angle_to(self.distance_to_fruit)

    @property
    def left(self) -> Vector2:
        """
        Vector2: The position to the left of the snake's head.
        """
        return Vector2(-self.direction.y, self.direction.x)

    @property
    def right(self) -> Vector2:
        """
        Vector2: The position to the right of the snake's head.
        """
        return Vector2(self.direction.y, -self.direction.x)

    @property
    def up(self) -> Vector2:
        """
        Vector2: The position above the snake's head.
        """
        return Vector2(-self.direction.x, -self.direction.y)

    @property
    def distance_to_wall(self) -> list[float]:
        """
        list[float]: The distance to the walls in the up, down, left, and right directions.
        """
        return [
            self.body[0].y,
            19 - self.body[0].y,
            self.body[0].x,
            19 - self.body[0].x,
        ]

    @property
    def distance_to_body(self) -> list[float]:
        """
        list[float]: The distance to the body in the up, down, left, and right directions.
        """
        return [
            self.body[1].y,
            19 - self.body[1].y,
            self.body[1].x,
            19 - self.body[1].x,
        ]

    @property
    def space_left(self) -> bool:
        """
        bool: True if there is space to the left of the snake's head, False otherwise.
        """
        return self.body[0] + self.left not in self.body

    @property
    def space_right(self) -> bool:
        """
        bool: True if there is space to the right of the snake's head, False otherwise.
        """
        return self.body[0] + self.right not in self.body

    @property
    def space_up(self) -> bool:
        """
        bool: True if there is space above the snake's head, False otherwise.
        """
        return self.body[0] + self.up not in self.body

    @property
    def space_down(self) -> bool:
        """
        bool: True if there is space below the snake's head, False otherwise.
        """
        return self.body[0] + self.direction not in self.body

    @property
    def vision(self) -> list[float]:
        """
        list[float]: The snake's vision of the surrounding environment.
        """
        return [
            self.space_left,
            self.space_right,
            self.space_up,
            self.space_down,
        ]

    @property
    def decision(self) -> int:
        """
        int: The decision made by the neural network based on the snake's vision.
        """
        return self.network.predict(self.vision)

    @property
    def steps(self) -> int:
        """
        int: The number of steps the snake has taken.
        """
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        """
        Sets the number of steps the snake has taken.

        Args:
            value (int): The number of steps taken.
        """
        self._steps = value
        self.fitness = self.score + self.life_time
        logging.debug(
            f"Setting the number of steps taken by the snake to {value} with updated fitness value."
        )
        logging.debug(
            f"Setting the fitness value of the snake to {self.fitness} with updated steps value."
        )

    @property
    def fitness(self) -> float:
        """
        float: The fitness value of the snake.
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        """
        Sets the fitness value of the snake.

        Args:
            value (float): The fitness value of the snake.
        """
        self._fitness = value
        logging.debug(
            f"Setting the fitness value of the snake to {value} with updated steps value."
        )

    @property
    def score(self) -> int:
        """
        int: The score of the snake.
        """
        return self._score

    @score.setter
    def score(self, value: int) -> None:
        """
        Sets the score of the snake.

        Args:
            value (int): The score of the snake.
        """
        self._score = value
        logging.debug(f"Setting the score of the snake to {value}.")

    @property
    def life_time(self) -> int:
        """
        int: The life time of the snake.
        """
        return self._life_time

    @life_time.setter
    def life_time(self, value: int) -> None:
        """
        Sets the life time of the snake.

        Args:
            value (int): The life time of the snake.

        Raises:
            ValueError: If the life time is negative.
        """
        if value < 0:
            raise ValueError("The life time of the snake cannot be negative.")
        self._life_time = value
        logging.debug(f"Setting the life time of the snake to {value}.")

    @property
    def body(self) -> list[Vector2]:
        """
        list[Vector2]: The body of the snake.
        """
        return self._body

    @body.setter
    def body(self, value: list[Vector2]) -> None:
        """
        Sets the body of the snake.

        Args:
            value (list[Vector2]): The body of the snake.
        """
        self._body = value
        logging.debug(f"Setting the body of the snake to {value}.")

    @property
    def network(self) -> NeuralNetwork:
        """
        NeuralNetwork: The neural network of the snake.
        """
        return self._network

    @network.setter
    def network(self, value: NeuralNetwork) -> None:
        """
        Sets the neural network of the snake.

        Args:
            value (NeuralNetwork): The neural network of the snake.
        """
        self._network = value
        logging.debug(f"Setting the neural network of the snake to {value}.")

    @property
    def hidden(self) -> int:
        """
        int: The number of hidden layers in the neural network.
        """
        return self._hidden

    @hidden.setter
    def hidden(self, value: int) -> None:
        """
        Sets the number of hidden layers in the neural network.

        Args:
            value (int): The number of hidden layers in the neural network.
        """
        self._hidden = value
        logging.debug(
            f"Setting the number of hidden layers in the neural network to {value}."
        )

    @property
    def learning_rate(self) -> float:
        """
        float: The learning rate of the neural network.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """
        Sets the learning rate of the neural network.

        Args:
            value (float): The learning rate of the neural network.
        """
        self._learning_rate = value
