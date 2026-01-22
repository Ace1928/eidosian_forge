import numpy as np
import random
from typing import List, Optional


class Tile:
    def __init__(self, x: int, y: int, value: int = 2) -> None:
        self.position = np.array([x, y], dtype=int)
        self.value = value
        self.death_on_impact = False
        self.already_increased = False

    def move_to(self, new_position: np.array) -> None:
        self.position = new_position

    def set_color(self) -> None:
        """Update the tile's color based on its value. This method is a placeholder and should be
        implemented to integrate with the specific UI framework being used."""
        pass

    def show(self) -> None:
        """Display the tile in the UI. This method is a placeholder and should integrate with the specific UI framework."""
        print(f"Tile at {self.position} with value {self.value}")

    def clone(self) -> "Tile":
        return Tile(
            self.position[0],
            self.position[1],
            self.value,
            self.death_on_impact,
            self.already_increased,
        )


class Player:
    def __init__(self, is_replay: bool = False) -> None:
        self.fitness: int = 0
        self.dead: bool = False
        self.score: int = 0
        self.tiles: List[Tile] = []
        self.empty_positions: List[np.array] = []
        self.move_direction: np.array = np.array([0, 0], dtype=int)
        self.moving_tiles: bool = False
        self.tile_moved: bool = False
        self.starting_positions: np.array = np.zeros((2, 3), dtype=int)

        self.fill_empty_positions()
        if not is_replay:
            self.add_new_tile()
            self.add_new_tile()
            self.set_starting_positions()

    def fill_empty_positions(self) -> None:
        self.empty_positions = [np.array([i, j]) for i in range(4) for j in range(4)]

    def set_empty_positions(self) -> None:
        self.empty_positions.clear()
        for i in range(4):
            for j in range(4):
                if self.get_value(i, j) == 0:
                    self.empty_positions.append(np.array([i, j]))

    def set_starting_positions(self) -> None:
        if len(self.tiles) >= 2:
            self.starting_positions[0, :] = np.append(
                self.tiles[0].position, self.tiles[0].value
            )
            self.starting_positions[1, :] = np.append(
                self.tiles[1].position, self.tiles[1].value
            )

    def add_new_tile(self, value: Optional[int] = None) -> None:
        if not self.empty_positions:
            return
        index = random.randint(0, len(self.empty_positions) - 1)
        position = self.empty_positions.pop(index)
        if value is None:
            value = 4 if random.random() < 0.1 else 2
        new_tile = Tile(position[0], position[1], value)
        new_tile.set_color()
        self.tiles.append(new_tile)

    def add_new_tile_not_random(self) -> None:
        if not self.empty_positions:
            return
        not_random_number = self.score + sum(
            tile.position[0] + tile.position[1] + i for i, tile in enumerate(self.tiles)
        )
        index = not_random_number % len(self.empty_positions)
        position = self.empty_positions.pop(index)
        value = 4 if not_random_number % 10 < 9 else 2
        new_tile = Tile(position[0], position[1], value)
        new_tile.set_color()
        self.tiles.append(new_tile)

    def show(self) -> None:
        for tile in sorted(self.tiles, key=lambda x: x.death_on_impact):
            tile.show()

    def move_tiles(self) -> None:
        self.tile_moved = False
        for tile in self.tiles:
            tile.already_increased = False
        if np.any(self.move_direction != 0):
            sorting_order = self.calculate_sorting_order()
            for order in sorting_order:
                for tile in self.tiles:
                    if np.array_equal(tile.position, order):
                        self.process_tile_movement(tile)

    def calculate_sorting_order(self) -> List[np.array]:
        sorting_vec = (
            np.array([3, 0])
            if self.move_direction[0] == 1
            else (
                np.array([0, 0])
                if self.move_direction[0] == -1
                else (
                    np.array([0, 3])
                    if self.move_direction[1] == 1
                    else np.array([0, 0])
                )
            )
        )
        vert = self.move_direction[1] != 0
        sorting_order = []
        for i in range(4):
            for j in range(4):
                temp = sorting_vec.copy()
                if vert:
                    temp[0] += j
                else:
                    temp[1] += j
                sorting_order.append(temp)
            sorting_vec -= self.move_direction
        return sorting_order

    def process_tile_movement(self, tile: Tile) -> None:
        move_to = tile.position + self.move_direction
        while self.is_position_empty(move_to):
            tile.move_to(move_to)
            move_to += self.move_direction
            self.tile_moved = True
        self.handle_potential_merge(tile, move_to)

    def is_position_empty(self, position: np.array) -> bool:
        return all(not np.array_equal(t.position, position) for t in self.tiles)

    def handle_potential_merge(self, tile: Tile, position: np.array) -> None:
        other = self.get_tile_at(position)
        if other and other.value == tile.value and not other.already_increased:
            tile.move_to(position)
            tile.death_on_impact = True
            other.already_increased = True
            other.value *= 2
            self.score += other.value
            other.set_color()
            self.tile_moved = True

    def get_tile_at(self, position: np.array) -> Optional[Tile]:
        for tile in self.tiles:
            if np.array_equal(tile.position, position):
                return tile
        return None

    def get_value(self, x: int, y: int) -> int:
        tile = self.get_tile_at(np.array([x, y]))
        return tile.value if tile else 0

    def move(self) -> None:
        if self.moving_tiles:
            for tile in self.tiles:
                tile.position += self.move_direction
            if self.done_moving():
                self.tiles = [tile for tile in self.tiles if not tile.death_on_impact]
                self.moving_tiles = False
                self.set_empty_positions()
                self.add_new_tile_not_random()

    def done_moving(self) -> bool:
        return all(not tile.death_on_impact for tile in self.tiles)

    def update(self) -> None:
        self.move()

    def set_tiles_from_history(self) -> None:
        self.tiles.clear()
        for i in range(2):
            pos = self.starting_positions[i, :2].astype(int)
            val = self.starting_positions[i, 2]
            tile = Tile(pos[0], pos[1], val)
            self.tiles.append(tile)
        self.remove_occupied_from_empty_positions()

    def remove_occupied_from_empty_positions(self) -> None:
        occupied_positions = [tile.position for tile in self.tiles]
        self.empty_positions = [
            pos
            for pos in self.empty_positions
            if not any(np.array_equal(pos, o_pos) for o_pos in occupied_positions)
        ]
