import random
import numpy as np
import pygame
import sys

pygame.display.init()


class Environment:
    def __init__(self):
        self.size = self.calculate_size()
        self.grid = np.full((self.size, self.size), None, dtype=object)
        self.resources = np.zeros((self.size, self.size), dtype=int)
        self.resource_access_count = np.zeros((self.size, self.size), dtype=int)
        self.populate_resources()

    def calculate_size(self) -> int:
        screen_info = pygame.display.Info()
        cell_size = 5
        return min(screen_info.current_w, screen_info.current_h) // cell_size

    def update_size(self):
        self.size = self.calculate_size()
        self.grid = np.full((self.size, self.size), None, dtype=object)
        self.resources = np.zeros((self.size, self.size), dtype=int)
        self.resource_access_count = np.zeros((self.size, self.size), dtype=int)

    def populate_resources(self):
        for _ in range(self.size * 2):
            x, y = self.get_random_position()
            self.resources[x, y] += 1

    def place_creature(self, creature, position: tuple[int, int]):
        self.grid[position] = creature

    def move_creature(
        self, old_position: tuple[int, int], new_position: tuple[int, int]
    ):
        self.grid[new_position], self.grid[old_position] = self.grid[old_position], None

    def get_random_position(self) -> tuple[int, int]:
        while True:
            position = random.randint(0, self.size - 1), random.randint(
                0, self.size - 1
            )
            if self.grid[position] is None:
                return position

    def display(self):
        grid_display = "\n".join(
            [
                "".join(["." if cell is None else "C" for cell in row])
                for row in self.grid
            ]
        )
        resources_display = "\n".join(map(str, self.resources))
        print(f"{grid_display}\n\nResources:\n{resources_display}\n")

    def update_resources(self, position: tuple[int, int]):
        x, y = position
        self.resource_access_count[x, y] += 1
        if self.resource_access_count[x, y] >= 5:
            self.resources[x, y] = 0
            self.resource_access_count[x, y] = 0
            new_x, new_y = self.get_random_position()
            self.resources[new_x, new_y] += 1

    def apply_disease(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y]:
                    creature = self.grid[x, y]
                    neighbors = sum(
                        1
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if 0 <= x + dx < self.size
                        and 0 <= y + dy < self.size
                        and self.grid[x + dx, y + dy]
                    )
                    if neighbors > 2 or self.resources[x, y] == 0:
                        creature.health -= 0.1


class Creature:
    def __init__(
        self,
        health: float,
        attack: float,
        speed: float,
        intelligence: float,
        strategy: str,
    ):
        self.health = health
        self.attack = attack
        self.speed = speed
        self.intelligence = intelligence
        self.strategy = strategy
        self.movement_energy_cost = 0.1 * self.speed

    def battle(self, other: "Creature") -> bool:
        while self.health > 0 and other.health > 0:
            other.health -= self.attack
            if other.health <= 0:
                return True
            self.health -= other.attack
        return self.health > 0

    def mutate(self) -> None:
        mutation_factor = 0.1
        self.health = max(1, self.health + random.uniform(-1, 1) * mutation_factor)
        self.attack = max(1, self.attack + random.uniform(-1, 1) * mutation_factor)
        self.speed = max(1, self.speed + random.uniform(-1, 1) * mutation_factor)
        self.intelligence = max(
            1, self.intelligence + random.uniform(-1, 1) * mutation_factor
        )
        if random.random() < 0.1:
            self.strategy = random.choice(
                ["battle", "flee", "group", "mate", "new_strategy"]
            )

    def __repr__(self) -> str:
        return (
            f"Creature(Health: {self.health}, Attack: {self.attack}, Speed: {self.speed}, "
            f"Intelligence: {self.intelligence}, Strategy: {self.strategy})"
        )

    def choose_move(
        self, current_position: tuple[int, int], environment: "Environment"
    ) -> tuple[int, int]:
        x, y = current_position
        best_move = (x, y)
        best_score = float("-inf")

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < environment.size
                and 0 <= ny < environment.size
                and environment.grid[nx, ny] is None
            ):
                score = (
                    random.uniform(0, 1) * self.intelligence
                    + environment.resources[nx, ny]
                )
                if score > best_score:
                    best_move = (nx, ny)
                    best_score = score

        return best_move

    def acquire_resources(
        self, position: tuple[int, int], environment: "Environment"
    ) -> None:
        x, y = position
        if environment.resources[x, y] > 0:
            self.health += environment.resources[x, y]
            environment.update_resources((x, y))


class Battle:
    def __init__(self, environment):
        self.environment = environment

    def fight(self, position1, position2):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if creature1 and creature2:
            winner = creature1.battle(creature2)
            self.environment.grid[position2 if winner else position1] = (
                creature1 if winner else creature2
            )

    def handle_encounter(self, position1, position2):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if not (creature1 and creature2):
            return

        strategy_actions = {
            "battle": self.fight,
            "flee": self.flee,
            "mate": self.mate,
            "group": self.group,
            "new_strategy": self.new_strategy,
        }

        for strategy in ["battle", "flee", "mate", "group", "new_strategy"]:
            if creature1.strategy == strategy or creature2.strategy == strategy:
                strategy_actions[strategy](creature1, creature2, position1, position2)
                break

    def flee(self, creature1, creature2, position1, position2):
        if creature1.strategy == "flee":
            new_position = creature1.choose_move(position1, self.environment)
            self.environment.move_creature(position1, new_position)
        if creature2.strategy == "flee":
            new_position = creature2.choose_move(position2, self.environment)
            self.environment.move_creature(position2, new_position)

    def mate(self, creature1, creature2, position1, position2):
        if creature1.strategy == "mate" and creature2.strategy == "mate":
            new_health = (creature1.health + creature2.health) / 2
            new_attack = (creature1.attack + creature2.attack) / 2
            new_speed = (creature1.speed + creature2.speed) / 2
            new_intelligence = (creature1.intelligence + creature2.intelligence) / 2
            new_strategy = random.choice(
                ["battle", "flee", "group", "mate", "new_strategy"]
            )
            new_creature = Creature(
                new_health, new_attack, new_speed, new_intelligence, new_strategy
            )
            self.environment.place_creature(
                new_creature, self.environment.get_random_position()
            )

    def group(self, creature1, creature2, position1, position2):
        if creature1.strategy == "group" and creature2.strategy == "group":
            new_position1 = creature1.choose_move(position1, self.environment)
            new_position2 = creature2.choose_move(position2, self.environment)
            if new_position1 == new_position2:
                self.environment.move_creature(position1, new_position1)
                self.environment.move_creature(position2, new_position2)

    def new_strategy(self, creature1, creature2, position1, position2):
        # Implement new complex behavior here
        pass


class Evolution:
    def __init__(self, environment):
        self.environment = environment

    def evolve(self):
        """Evolve creatures in the environment by mutating and repositioning them."""
        new_creatures = [
            self.environment.grid[x, y].mutate() or self.environment.grid[x, y]
            for x in range(self.environment.size)
            for y in range(self.environment.size)
            if self.environment.grid[x, y]
        ]

        self.environment.grid.fill(None)

        for creature in new_creatures:
            self.environment.place_creature(
                creature, self.environment.get_random_position()
            )


class UserInteraction:
    def __init__(self, environment, evolution, battle):
        self.environment = Environment()
        self.evolution = Evolution(self.environment)
        self.battle = Battle(self.environment)

    def add_creature(self, health, attack, speed, intelligence, strategy, position):
        creature = Creature(health, attack, speed, intelligence, strategy)
        self.environment.place_creature(creature, position)

    def initialize_environment(self):
        """Initialize or reset the environment grid and resources."""
        self.environment.grid = np.full(
            (self.environment.size, self.environment.size), None, dtype=object
        )
        self.environment.resources = np.zeros(
            (self.environment.size, self.environment.size), dtype=int
        )
        self.environment.resource_access_count = np.zeros(
            (self.environment.size, self.environment.size), dtype=int
        )
        self.environment.populate_resources()

    def handle_events(self, screen, cell_size):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                self.environment.update_size()
                screen = pygame.display.set_mode(
                    (
                        self.environment.size * cell_size,
                        self.environment.size * cell_size,
                    ),
                    pygame.RESIZABLE,
                )
                self.initialize_environment()

    def draw_grid(self, screen, cell_size):
        """Draw the grid with creatures and resources."""
        for x in range(self.environment.size):
            for y in range(self.environment.size):
                if self.environment.grid[x, y]:
                    pygame.draw.rect(
                        screen,
                        (0, 255, 0),
                        (x * cell_size, y * cell_size, cell_size, cell_size),
                    )
                elif self.environment.resources[x, y] > 0:
                    pygame.draw.rect(
                        screen,
                        (0, 0, 255),
                        (x * cell_size, y * cell_size, cell_size, cell_size),
                    )

    def move_creatures(self):
        """Move creatures based on their strategies."""
        for x in range(self.environment.size):
            for y in range(self.environment.size):
                if self.environment.grid[x, y]:
                    creature = self.environment.grid[x, y]
                    new_position = creature.choose_move((x, y), self.environment)
                    if new_position != (x, y):
                        self.environment.move_creature((x, y), new_position)

    def handle_encounters(self):
        """Handle encounters between creatures."""
        for x in range(self.environment.size):
            for y in range(self.environment.size):
                if self.environment.grid[x, y]:
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < self.environment.size
                            and 0 <= ny < self.environment.size
                        ):
                            if self.environment.grid[nx, ny]:
                                self.battle.handle_encounter((x, y), (nx, ny))

    def run_game(self, generations=10):
        pygame.init()
        cell_size = 5
        self.environment.update_size()
        self.initialize_environment()

        screen = pygame.display.set_mode(
            (self.environment.size * cell_size, self.environment.size * cell_size),
            pygame.RESIZABLE,
        )
        clock = pygame.time.Clock()

        # Add initial population of 100 creatures
        for _ in range(100):
            health = random.randint(5, 15)
            attack = random.randint(1, 5)
            speed = random.randint(1, 5)
            intelligence = random.randint(1, 5)
            strategy = random.choice(
                ["battle", "flee", "group", "mate", "new_strategy"]
            )
            position = self.environment.get_random_position()
            self.add_creature(health, attack, speed, intelligence, strategy, position)

        while True:
            self.handle_events(screen, cell_size)
            screen.fill((0, 0, 0))

            for generation in range(generations):
                self.evolution.evolve()
                self.environment.apply_disease()
                self.move_creatures()
                self.handle_encounters()
                self.draw_grid(screen, cell_size)

            pygame.display.flip()
            clock.tick(5)  # Adjust the speed as needed


if __name__ == "__main__":
    environment = Environment()
    evolution = Evolution(environment)
    battle = Battle(environment)
    interaction = UserInteraction(environment, evolution, battle)

    # Run the game indefinitely
    interaction.run_game()
